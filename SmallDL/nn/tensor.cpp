#include "m_operations.hpp"
#include "tensor.hpp"
#include "memory.hpp"
#include <algorithm>
#include <iostream>

Tensor::Tensor(Matrix m, bool req_grad, mem::mem_chunk* memory)
    : data(std::move(m)), grad(data.get_rows(), data.get_columns(), memory), requires_grad(req_grad), memory(memory), operation(""), backward_operation(nullptr) {
  ops::scale_(grad, 0.0f, memory);
}

Tensor::Tensor(unsigned int rows, unsigned int columns, bool req_grad, mem::mem_chunk* memory)
    : data(rows, columns, memory), grad(rows, columns, memory), requires_grad(req_grad), operation(""), memory(memory), backward_operation(nullptr) {
  ops::scale_(grad, 0.0f, memory);
}

void Tensor::backward() {
  if (operation == "mse_loss") {
    ops::scale_(grad, 0.0f, memory);
    ops::add_scalar_(grad, 1.0f, memory);
  }
  if (backward_operation)
    backward_operation();
  for (const auto &i : input) {
    if (i->requires_grad)
      i->backward();
  }
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
  Matrix m = ops::matmul(a->data, b->data, a->memory);
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad, a->memory);
  out->operation = "matmul";
  out->input.push_back(a);
  out->input.push_back(b);
  if (req_grad) {
    out->backward_operation = [a, b, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        Matrix t_b = ops::transpose(b->data, a->memory);
        Matrix g_a = ops::matmul(out->grad, t_b, a->memory);
        ops::add_(a->grad, g_a, a->memory);
      }
      if (b->requires_grad) {
        Matrix t_a = ops::transpose(a->data, a->memory);
        Matrix g_b = ops::matmul(t_a, out->grad, a->memory);
        ops::add_(b->grad, g_b, a->memory);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
  Matrix m(a->data, a->memory);
  ops::add_bias_(m, b->data, a->memory);
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad, a->memory);
  out->operation = "add_bias";
  out->input.push_back(a);
  out->input.push_back(b);
  if (req_grad) {
    out->backward_operation = [a, b, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        ops::add_(a->grad, out->grad, a->memory);
      }
      if (b->requires_grad) {
        unsigned int out_rows = out->grad.get_rows();
        unsigned int b_cols = b->data.get_columns();
        Matrix g_b(1, b_cols, a->memory);
        float *gb_ptr = g_b.data();
        const float *out_g_ptr = out->grad.data();
        g_b.last_event = a->memory->q->submit([&](sycl::handler &h){
          h.depends_on(out->grad.last_event);   
          h.parallel_for(sycl::range<1>(b_cols), [=](sycl::id<1> id) {
            unsigned int c = id[0];
            float sum = 0.0f;
            for (unsigned int r = 0; r < out_rows; ++r) {
              sum += out_g_ptr[r * b_cols + c];
            }
            gb_ptr[c] = sum;
          });
        });
        ops::add_(b->grad, g_b, a->memory);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
  Matrix m = ops::relu_forward(a->data, a->memory);
  bool req_grad = a->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad, a->memory);
  out->operation = "relu";
  out->input.push_back(a);
  if (req_grad) {
    out->backward_operation = [a, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        Matrix g_a = ops::relu_backward(a->data, out->grad, a->memory);
        ops::add_(a->grad, g_a, a->memory);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
  float loss = ops::mse_loss_forward(pred->data, target->data, pred->memory);
  Matrix loss_m(1, 1, pred->memory);
  loss_m.last_event = pred->memory->q->memcpy(loss_m.data(), &loss, sizeof(float));
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(loss_m), true, pred->memory);
  out->operation = "mse_loss";
  out->input.push_back(pred);
  out->input.push_back(target);
  mem::mem_chunk* cur_mem = pred->memory;
  out->backward_operation = [pred, target, cur_mem, weak_out = out->weak_from_this()]() {
    auto out = weak_out.lock();
    if (!out)
      return;
    if (pred->requires_grad) {
      Matrix g_pred = ops::mse_loss_backward(pred->data, target->data, cur_mem);
      ops::add_(pred->grad, g_pred, cur_mem);
    }
  };
  return out;
}

