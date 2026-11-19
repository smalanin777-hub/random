#ifndef LAYER_HPP
#define LAYER_HPP

#include "graph.hpp"

struct Layer {
  Tensor weight;
  Tensor bias;
  Tensor forward(Graph &g, Tensor a) const;
};

Layer make_layer(Graph &g, unsigned int in, unsigned int out);

#endif
