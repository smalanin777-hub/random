#include "layer.hpp"
#include "t_operations.hpp"

Layer make_layer(Graph &g, unsigned int in, unsigned int out) { return Layer{g.param(in, out), g.param(1, out)}; }
Tensor Layer::forward(Graph &g, Tensor a) const {
  Tensor temp = g.emit<MatMul>(a, weight);
  return g.emit<AddBias>(temp, bias);
}
