#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "constants.hpp"
#include "matrices.hpp"
#include "reflection.hpp"
#include "tensor.hpp"
#include <vector>

struct Node {
  unsigned char op_id = 0;
  unsigned int inputs[3] = {};
  unsigned int output = 0;
  bool is_dead = false;
};

struct Graph {
  sycl::queue q;
  mem::ArenaAlloc g_alloc;
  mem::ArenaAlloc t_alloc;
  unsigned long compiled_temp_offset = 0;
  std::vector<TensorStorage> tensors;
  std::vector<Node> nodes;
  Graph(unsigned long global_size = pow2_v<24>, unsigned long temp_size = pow2_v<26>);
  Tensor param(unsigned int rows, unsigned int cols);
  Tensor input(unsigned int rows, unsigned int cols);
  template <typename Operation, typename... Inputs> Tensor emit(Inputs... in) {
    static_assert(sizeof...(Inputs) <= 3, "Max 3 inputs per node supported");
    unsigned char id = reflect::get_op_id<Operation>();
    Shape out_shape = Operation::infer_shape(in.shape...);
    bool req_grad = (in.requires_grad || ...);
    unsigned int out_id = static_cast<unsigned int>(tensors.size());
    tensors.push_back(TensorStorage{Matrix{}, Matrix{}, out_shape, req_grad, false, false});
    Node node{};
    node.op_id = id;
    unsigned int in_ids[] = {in.id...};
    for (int i = 0; i != sizeof...(Inputs); ++i)
      node.inputs[i] = in_ids[i];
    node.output = out_id;
    nodes.push_back(node);
    return Tensor{out_id, out_shape, req_grad};
  }
  void compile();
  void forward();
  void backward();
  void reset_temp();
  void dump() const;
};

#endif
