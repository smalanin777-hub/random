#ifndef T_OPERATIONS_HPP
#define T_OPERATIONS_HPP

#include "graph.hpp"
#include "reflection.hpp"
#include <stdexcept>

struct MatMul {
  static Shape infer_shape(Shape a, Shape b);
  static void forward(void *graph, unsigned int node_id);
  static void backward(void *graph, unsigned int node_id);
  static constexpr reflect::OpMetaData meta() { return {"MatMul", reflect::OP_CAN_ACCEPT_EPI, 2, &forward, &backward}; }
};

struct AddBias {
  static Shape infer_shape(Shape a, Shape bias);
  static void forward(void *graph, unsigned int node_id);
  static void backward(void *graph, unsigned int node_id);
  static constexpr reflect::OpMetaData meta() {
    return {"AddBias", reflect::OP_ELEMENTWISE | reflect::OP_CAN_BE_EPI | reflect::OP_INPLACE_SAFE, 2, &forward, &backward};
  }
};

struct ReLU {
  static Shape infer_shape(Shape a);
  static void forward(void *graph, unsigned int node_id);
  static void backward(void *graph, unsigned int node_id);
  static constexpr reflect::OpMetaData meta() {
    return {"ReLU", reflect::OP_ELEMENTWISE | reflect::OP_CAN_BE_EPI | reflect::OP_INPLACE_SAFE, 1, &forward, &backward};
  }
};
struct MSELoss {
  static Shape infer_shape(Shape a, Shape target);
  static void forward(void *graph, unsigned int node_id);
  static void backward(void *graph, unsigned int node_id);
  static constexpr reflect::OpMetaData meta() { return {"MSELoss", reflect::OP_NONE, 2, &forward, &backward}; }
};

struct FusedMatMulBiasReLU {
  static Shape infer_shape(Shape a, Shape b, Shape bias);
  static void forward(void *graph, unsigned int node_id);
  static void backward(void *graph, unsigned int node_id);
  static constexpr reflect::OpMetaData meta() { return {"FusedMatMulBiasReLU", reflect::OP_NONE, 3, &forward, &backward}; }
};

#endif
