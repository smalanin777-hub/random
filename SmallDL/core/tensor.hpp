#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "matrices.hpp"
#include "m_operations.hpp"
#include "memory.hpp"
#include <functional>
#include <string>
#include <vector>

struct Tensor : public std::enable_shared_from_this<Tensor> {
  Matrix data;
  Matrix grad;
  mem::mem_chunk* memory;
  bool requires_grad;
  std::string operation;
  std::vector<std::shared_ptr<Tensor>> input;
  std::function<void()> backward_operation;
  Tensor(Matrix m, bool req_grad = false, mem::mem_chunk* memory = &global_chunk);
  Tensor(unsigned int rows, unsigned int columns, bool req_grad = false, mem::mem_chunk* memory = &global_chunk);
  void backward();
};

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);

struct Layer {
  std::shared_ptr<Tensor> W;
  std::shared_ptr<Tensor> b;
  Layer(unsigned int in, unsigned int out, mem::mem_chunk* memory = &global_chunk) {
    W = std::make_shared<Tensor>(in, out, true, memory);
    b = std::make_shared<Tensor>(1, out, true, memory);
    ops::generate_matrix(W->data, memory);
    ops::scale_(b->data, 0.0f, memory);
  }
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> a) { 
    return add_bias(matmul(a, W), b); 
  }
};

#endif
