#include"tensor.hpp"
#include<algorithm>
#include<iostream>

Tensor::Tensor(Matrix m, bool req_grad) : data(std::move(m)), grad(data.get_rows(), data.get_columns()), requires_grad(req_grad), operation(""), backward_operation(nullptr) {
  grad *= 0.0f; 
}
Tensor::Tensor(unsigned int rows, unsigned int columns, sycl::queue q, bool req_grad) : data(rows, columns, q), grad(rows, columns, q), requires_grad(req_grad), operation(""), backward_operation(nullptr) {
  grad *= 0.0f;
}
void Tensor::backward(){
  if(operation == "mse_loss"){
    grad *= 0.0f;
    grad += 1.0f;
  }
  if(backward_operation)
    backward_operation();
  for(const auto& i : input){
    if(i->requires_grad)
      i->backward();
  }
}
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
  Matrix m = a->data * b->data;
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "matmul";
  out->input.push_back(a);
  out->input.push_back(b);
  if(req_grad){
    out->backward_operation = [a, b, out](){
      if(a->requires_grad){
        Matrix t_b = b->data.transpose();
        Matrix g_a = out->grad * t_b;
        a->grad = a->grad + g_a; 
      }
      if(b->requires_grad){
        Matrix t_a = a->data.transpose();
        Matrix g_b = t_a * out->grad;
        b->grad = b->grad + g_b;
      }
    };
  } 
  return out;
}
std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
  Matrix m = a->data;
  m.add_bias(b->data);
  bool req_grad = a->requires_grad || b->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "add_bias";
  out->input.push_back(a);
  out->input.push_back(b);
  if(req_grad){
    out->backward_operation = [a, b, out](){
      if(a->requires_grad) {
        a->grad = a->grad + out->grad;
      }
      if(b->requires_grad){
        unsigned int out_rows = out->grad.get_rows();
        unsigned int b_cols = b->data.get_columns();
        sycl::queue q = global_q;
        Matrix g_b(1, b_cols, q);
        float* gb_ptr = g_b.data();
        const float* out_g_ptr = out->grad.data();
        q.parallel_for(sycl::range<1>(b_cols), [=](sycl::id<1> id){
          unsigned int c = id[0];
          float sum = 0.0f;
          for(unsigned int r = 0; r < out_rows; ++r){
            sum += out_g_ptr[r * b_cols + c];
          }
          gb_ptr[c] = sum;
        }).wait();
        b->grad = b->grad + g_b; 
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a){
  Matrix m = a->data.relu_forward();
  bool req_grad = a->requires_grad;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(m), req_grad);
  out->operation = "relu";
  out->input.push_back(a);
  if(req_grad){
    out->backward_operation = [a, out](){
      if(a->requires_grad){
        Matrix g_a = a->data.relu_backward(out->grad);
        a->grad = a->grad + g_a;
      }
    };
  }
  return out;
}
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target){
  float loss = pred->data.mse_loss_forward(target->data);
  Matrix loss_m(1, 1, global_q);
  loss_m.data()[0] = loss;
  std::shared_ptr<Tensor> out = std::make_shared<Tensor>(std::move(loss_m), true);
  out->operation = "mse_loss";
  out->input.push_back(pred);
  out->input.push_back(target);
  out->backward_operation = [pred, target, out](){
    if(pred->requires_grad){
      Matrix g_pred = pred->data.mse_loss_backward(target->data);
      pred->grad = pred->grad + g_pred;
    }
  };
  return out;
}











