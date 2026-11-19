#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "reflection.hpp"

struct Tensor {
  unsigned int id = 0;
  Shape shape{};
  bool requires_grad = false;
};

struct TensorStorage {
  Matrix data;
  Matrix grad;
  Shape shape{};
  bool requires_grad = false;
  bool is_persistent = false;
  bool is_dead = false;
};

#endif
