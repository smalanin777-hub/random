#include "m_operations.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <iostream>

Tensor::Tensor(Matrix m, bool req_grad)
    : data(std::move(m)), grad(data.get_rows(), data.get_columns()), requires_grad(req_grad), operation(""), backward_operation(nullptr) {
  ops::scale_(grad, 0.0f);
}

Tensor::Tensor(unsigned int rows, unsigned int columns, sycl::queue q, bool req_grad)
    : data(rows, columns, q), grad(rows, columns, q), requires_grad(req_grad), operation(""), backward_operation(nullptr) {
  ops::scale_(grad, 0.0f);
}

void Tensor::backward() {
  if (operation == "mse_loss") {
    ops::scale_(grad, 0.0f);
    ops::add_scalar_(grad, 1.0f);
  }
  if (backward_operation)
    backward_operation();
  for (const auto &i : input) {
    if (i->requires_grad)
      i->backward();
  }
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
  Matrix m = ops::matmul(a->data, b->data);
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "matmul";
  out->input.push_back(a);
  out->input.push_back(b);
  if (req_grad) {
    out->backward_operation = [a, b, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        Matrix t_b = ops::transpose(b->data);
        Matrix g_a = ops::matmul(out->grad, t_b);
        ops::add_(a->grad, g_a);
      }
      if (b->requires_grad) {
        Matrix t_a = ops::transpose(a->data);
        Matrix g_b = ops::matmul(t_a, out->grad);
        ops::add_(b->grad, g_b);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
  Matrix m(a->data);
  ops::add_bias_(m, b->data);
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "add_bias";
  out->input.push_back(a);
  out->input.push_back(b);
  if (req_grad) {
    out->backward_operation = [a, b, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        ops::add_(a->grad, out->grad);
      }
      if (b->requires_grad) {
        unsigned int out_rows = out->grad.get_rows();
        unsigned int b_cols = b->data.get_columns();
        sycl::queue q = global_q;
        Matrix g_b(1, b_cols, q);
        float *gb_ptr = g_b.data();
        const float *out_g_ptr = out->grad.data();
        g_b.last_event = q.submit([&](sycl::handler &h){
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
        ops::add_(b->grad, g_b);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
  Matrix m = ops::relu_forward(a->data);
  bool req_grad = a->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "relu";
  out->input.push_back(a);
  if (req_grad) {
    out->backward_operation = [a, weak_out = out->weak_from_this()]() {
      auto out = weak_out.lock();
      if (!out)
        return;
      if (a->requires_grad) {
        Matrix g_a = ops::relu_backward(a->data, out->grad);
        ops::add_(a->grad, g_a);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
  float loss = ops::mse_loss_forward(pred->data, target->data);
  Matrix loss_m(1, 1, global_q);
  loss_m.data()[0] = loss;
  loss_m.last_event = loss_m.q.ext_oneapi_submit_barrier();
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(loss_m), true);
  out->operation = "mse_loss";
  out->input.push_back(pred);
  out->input.push_back(target);
  out->backward_operation = [pred, target, weak_out = out->weak_from_this()]() {
    auto out = weak_out.lock();
    if (!out)
      return;
    if (pred->requires_grad) {
      Matrix g_pred = ops::mse_loss_backward(pred->data, target->data);
      ops::add_(pred->grad, g_pred);
    }
  };
  return out;
}
