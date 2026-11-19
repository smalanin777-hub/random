#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "graph.hpp"
#include "m_operations.hpp"
#include "reflection.hpp"
#include "t_operations.hpp"
#include <vector>

struct GD {
private:
  std::vector<unsigned int> param_ids;
  float step_size;

public:
  template <typename Model> GD(Model &model, float step_size) : step_size(step_size) {
    reflect::bypass<Tensor>(model, [&](Tensor &t) {
      if (t.requires_grad)
        param_ids.push_back(t.id);
    });
  }
  void zero_grad(Graph &g) {
    for (unsigned int id : param_ids)
      ops::scale_(g.tensors[id].grad, 0.0f, g.q, g.t_alloc.events);
  }
  void step(Graph &g) {
    for (unsigned int id : param_ids) {
      TensorStorage &t = g.tensors[id];
      if (t.requires_grad)
        ops::sgd_step_(t.data, t.grad, step_size, g.q, g.t_alloc.events);
    }
  }
  void set_step(float new_step) { step_size = new_step; }
};
#endif
