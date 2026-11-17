#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include "constants.hpp"

struct Layer{
  std::shared_ptr<Tensor> a;
  std::shared_ptr<Tensor> b;
  Layer(unsigned int in, unsigned int out, mem::ArenaAlloc* memory = nullptr);
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> t);
};

#endif
