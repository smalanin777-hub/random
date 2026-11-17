#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "m_operations.hpp"
#include <functional>
#include <string>
#include <vector>

struct Tensor : public std::enable_shared_from_this<Tensor> {
  Matrix data;
  Matrix grad;
  mem::ArenaAlloc* memory;
  bool requires_grad;
  std::string operation;
  std::vector<std::shared_ptr<Tensor>> input;
  std::function<void()> backward_operation;
  Tensor(Matrix m, bool req_grad = false, mem::ArenaAlloc* memory = nullptr);
  Tensor(unsigned int rows, unsigned int columns, bool req_grad = false, mem::ArenaAlloc* memory = nullptr);
  void backward();
};

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> add_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);


#endif
