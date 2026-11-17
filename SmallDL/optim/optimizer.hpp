#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"
#include <vector>

struct GD {
private:
  std::vector<std::shared_ptr<Tensor>> params;
  float step_size;

public:
  GD(std::vector<std::shared_ptr<Tensor>> pars, float step_size) : params(pars), step_size(step_size) {}

  void zero_grad() {
    for (auto &i : params)
      ops::scale_(i->grad, 0.0f, i->memory);
  }
  void step(mem::ArenaAlloc* memory) {
    for (auto &i : params) {
      if (i->requires_grad) {
        Matrix step_m = ops::scale(i->grad, step_size, memory);
        ops::scale_(step_m, -1.0f, memory);
        ops::add_(i->data, step_m, i->memory);
      }
    }
  }
  void set_step(float new_step) {
    step_size = new_step;
  }
};
#endif
