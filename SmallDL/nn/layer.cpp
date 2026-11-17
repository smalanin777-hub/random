#include "layer.hpp"

Layer::Layer(unsigned int in, unsigned int out, mem::ArenaAlloc *memory) {
  a = std::make_shared<Tensor>(in, out, true, memory);
  b = std::make_shared<Tensor>(1, out, true, memory);
  ops::generate_matrix(a->data, memory);
  ops::scale_(b->data, 0.0f, memory);
}
std::shared_ptr<Tensor> Layer::forward(std::shared_ptr<Tensor> t) { return add_bias(matmul(t, a), b); }
