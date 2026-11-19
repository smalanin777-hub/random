#include "t_operations.hpp"
#include "m_operations.hpp"

Shape MatMul::infer_shape(Shape a, Shape b) {
  if (a.cols != b.rows)
    throw std::invalid_argument("MatMulOp: inner dimensions mismatch (a.cols != b.rows)");
  return {a.rows, b.cols};
}

Shape ReLU::infer_shape(Shape a) { return a; }

Shape AddBias::infer_shape(Shape a, Shape bias) {
  if (bias.rows != 1 || bias.cols != a.cols)
    throw std::invalid_argument("AddBiasOp: bias shape must be [1, a.cols]");
  return a;
}

Shape MSELoss::infer_shape(Shape pred, Shape target) {
  if (pred.rows != target.rows || pred.cols != target.cols)
    throw std::invalid_argument("MSELossOp: pred and target shapes mismatch");
  return {1, 1};
}

Shape FusedMatMulBiasReLU::infer_shape(Shape a, Shape b, Shape bias) {
  if (a.cols != b.rows)
    throw std::invalid_argument("FusedMatMulBiasReluOp: inner dimensions mismatch");
  if (bias.rows != 1 || bias.cols != b.cols)
    throw std::invalid_argument("FusedMatMulBiasReluOp: bias shape mismatch");
  return {a.rows, b.cols};
}

void MatMul::forward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  const TensorStorage &a = g->tensors[node.inputs[0]];
  const TensorStorage &b = g->tensors[node.inputs[1]];
  TensorStorage &out = g->tensors[node.output];
  ops::matmul(a.data, b.data, out.data, g->q, g->t_alloc.events);
}

void MatMul::backward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  TensorStorage &a = g->tensors[node.inputs[0]];
  TensorStorage &b = g->tensors[node.inputs[1]];
  const TensorStorage &out = g->tensors[node.output];
  if (a.requires_grad) {
    Matrix g_a(a.shape.rows, a.shape.cols, &g->t_alloc);
    ops::matmul_transpose(out.grad, b.data, g_a, g->q, g->t_alloc.events);
    ops::add_(a.grad, g_a, g->q, g->t_alloc.events);
  }
  if (b.requires_grad) {
    Matrix g_b(b.shape.rows, b.shape.cols, &g->t_alloc);
    ops::matmul_transpose(out.grad, a.data, g_b, g->q, g->t_alloc.events);
    ops::add_(b.grad, g_b, g->q, g->t_alloc.events);
  }
}

void AddBias::forward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  const TensorStorage &a = g->tensors[node.inputs[0]];
  const TensorStorage &bias = g->tensors[node.inputs[1]];
  TensorStorage &out = g->tensors[node.output];
  unsigned long bytes = static_cast<unsigned long>(a.shape.rows) * a.shape.cols * sizeof(float);
  g->t_alloc.events.push_back(g->q.memcpy(out.data.data(), a.data.data(), bytes));
  ops::add_bias_(out.data, bias.data, g->q, g->t_alloc.events);
}

void AddBias::backward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  TensorStorage &a = g->tensors[node.inputs[0]];
  TensorStorage &bias = g->tensors[node.inputs[1]];
  const TensorStorage &out = g->tensors[node.output];
  if (a.requires_grad)
    ops::add_(a.grad, out.grad, g->q, g->t_alloc.events);
  if (bias.requires_grad) {
    Matrix g_b(1, bias.shape.cols, &g->t_alloc);
    ops::sum_cols(out.grad, g_b, g->q, g->t_alloc.events);
    ops::add_(bias.grad, g_b, g->q, g->t_alloc.events);
  }
}

void ReLU::forward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  const TensorStorage &in = g->tensors[node.inputs[0]];
  TensorStorage &out = g->tensors[node.output];
  ops::relu_forward(in.data, out.data, g->q, g->t_alloc.events);
}

void ReLU::backward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  TensorStorage &in = g->tensors[node.inputs[0]];
  const TensorStorage &out = g->tensors[node.output];
  if (in.requires_grad) {
    Matrix g_in(in.shape.rows, in.shape.cols, &g->t_alloc);
    ops::relu_backward(in.data, out.grad, g_in, g->q, g->t_alloc.events);
    ops::add_(in.grad, g_in, g->q, g->t_alloc.events);
  }
}

void MSELoss::forward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  const TensorStorage &pred = g->tensors[node.inputs[0]];
  const TensorStorage &target = g->tensors[node.inputs[1]];
  TensorStorage &out = g->tensors[node.output];
  ops::mse_loss_forward(pred.data, target.data, out.data, g->q, g->t_alloc.events);
}

void MSELoss::backward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  TensorStorage &pred = g->tensors[node.inputs[0]];
  const TensorStorage &target = g->tensors[node.inputs[1]];
  if (pred.requires_grad) {
    Matrix g_pred(pred.shape.rows, pred.shape.cols, &g->t_alloc);
    ops::mse_loss_backward(pred.data, target.data, g_pred, g->q, g->t_alloc.events);
    ops::add_(pred.grad, g_pred, g->q, g->t_alloc.events);
  }
}

void FusedMatMulBiasReLU::forward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  const TensorStorage &a = g->tensors[node.inputs[0]];
  const TensorStorage &b = g->tensors[node.inputs[1]];
  const TensorStorage &bias = g->tensors[node.inputs[2]];
  TensorStorage &out = g->tensors[node.output];
  ops::matmul_bias_relu(a.data, b.data, bias.data, out.data, g->q, g->t_alloc.events);
}

void FusedMatMulBiasReLU::backward(void *graph, unsigned int node_id) {
  Graph *g = static_cast<Graph *>(graph);
  const Node &node = g->nodes[node_id];
  TensorStorage &a = g->tensors[node.inputs[0]];
  TensorStorage &b = g->tensors[node.inputs[1]];
  TensorStorage &bias = g->tensors[node.inputs[2]];
  const TensorStorage &out = g->tensors[node.output];
  Matrix d_z(out.shape.rows, out.shape.cols, &g->t_alloc);
  ops::relu_backward(out.data, out.grad, d_z, g->q, g->t_alloc.events);
  if (bias.requires_grad) {
    Matrix g_b(1, bias.shape.cols, &g->t_alloc);
    ops::sum_cols(d_z, g_b, g->q, g->t_alloc.events);
    ops::add_(bias.grad, g_b, g->q, g->t_alloc.events);
  }
  if (a.requires_grad) {
    Matrix t_b(b.shape.cols, b.shape.rows, &g->t_alloc);
    Matrix g_a(a.shape.rows, a.shape.cols, &g->t_alloc);
    ops::transpose(b.data, t_b, g->q, g->t_alloc.events);
    ops::matmul(d_z, t_b, g_a, g->q, g->t_alloc.events);
    ops::add_(a.grad, g_a, g->q, g->t_alloc.events);
  }
  if (b.requires_grad) {
    Matrix t_a(a.shape.cols, a.shape.rows, &g->t_alloc);
    Matrix g_b(b.shape.rows, b.shape.cols, &g->t_alloc);
    ops::transpose(a.data, t_a, g->q, g->t_alloc.events);
    ops::matmul(t_a, d_z, g_b, g->q, g->t_alloc.events);
    ops::add_(b.grad, g_b, g->q, g->t_alloc.events);
  }
}
