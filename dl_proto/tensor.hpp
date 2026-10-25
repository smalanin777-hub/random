#ifndef TENSOR_HPP
#define TENSOR_HPP

#include"matrices.hpp"
#include<memory>
#include<vector>
#include<string>
#include<functional>

struct Tensor : public std::enable_shared_from_this<Tensor> {
  Matrix data;
  Matrix grad;
  bool requires_grad;
  std::string operation;
  std::vector<std::shared_ptr<Tensor>> input;
  std::function<void()> backward_operation;
  Tensor(Matrix m, bool req_grad = false);
  Tensor(unsigned int rows, unsigned int columns, sycl::queue q = global_q, bool req_grad = false);
  void backward();
};

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);

struct Layer{
  std::shared_ptr<Tensor> W;
  std::shared_ptr<Tensor> b;
  Layer(unsigned int in, unsigned int out, sycl::queue q = global_q){
    W = std::make_shared<Tensor>(in, out, q, true);
    b = std::make_shared<Tensor>(1, out, q, true);
    W->data.generate_matrix();
    b->data *= 0.0f;
  }
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> a){
    return add_bias(matmul(a, W), b);
  }
};

#endif
