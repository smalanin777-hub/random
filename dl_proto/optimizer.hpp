#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include"tensor.hpp"
#include<vector>
#include<memory>

struct GD{
private:
  std::vector<std::shared_ptr<Tensor>> params;
  float step_size;
public:
  GD(std::vector<std::shared_ptr<Tensor>> pars, float step_size) : params(pars), step_size(step_size) {}
  void zero_grad(){
    for(auto& i : params)
      i->grad *= 0.0f; 
  }
  void step(){
    for(auto& i : params){
      if(i->requires_grad){
        Matrix step_m = i->grad * step_size;
        i->data = i->data + (step_m * -1.0f); 
      }
    }
  }
  void set_step(float new_step){
    step_size = new_step;
  }
};

#endif
