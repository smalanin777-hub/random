#include "graph.hpp"
#include "m_operations.hpp"
#include "t_operations.hpp"

Graph::Graph(unsigned long global_size, unsigned long temp_size)
    : q(sycl::gpu_selector_v), g_alloc(global_size, q), t_alloc(temp_size, q) {}

Tensor Graph::param(unsigned int rows, unsigned int cols) {
  unsigned int id = static_cast<unsigned int>(tensors.size());
  Matrix d(rows, cols, &g_alloc);
  Matrix g(rows, cols, &g_alloc);
  ops::generate_matrix(d, q, g_alloc.events);
  ops::scale_(g, 0.0f, q, g_alloc.events);
  tensors.push_back(TensorStorage{std::move(d), std::move(g), {rows, cols}, true, true, false});
  return Tensor{id, {rows, cols}, true};
}

Tensor Graph::input(unsigned int rows, unsigned int cols) {
  unsigned int id = static_cast<unsigned int>(tensors.size());
  Matrix d(rows, cols, &g_alloc);
  tensors.push_back(TensorStorage{std::move(d), Matrix{}, {rows, cols}, false, true, false});
  return Tensor{id, {rows, cols}, false};
}

void Graph::compile() {
  const unsigned char id_matmul = reflect::get_op_id<MatMul>();
  const unsigned char id_bias = reflect::get_op_id<AddBias>();
  const unsigned char id_relu = reflect::get_op_id<ReLU>();
  const unsigned char id_fused = reflect::get_op_id<FusedMatMulBiasReLU>();

  std::vector<unsigned int> use_count(tensors.size(), 0);
  for (const auto &node : nodes) {
    unsigned char n = reflect::get_op_meta(node.op_id).arguments;
    for (unsigned char i = 0; i < n; ++i)
      use_count[node.inputs[i]]++;
  }

  for (unsigned int i = 0; i < nodes.size() - 2; ++i) {
    Node &n0 = nodes[i];
    Node &n1 = nodes[i + 1];
    Node &n2 = nodes[i + 2];
    if (!n0.is_dead && n0.op_id == id_matmul && !n1.is_dead && n1.op_id == id_bias && n1.inputs[0] == n0.output &&
        use_count[n0.output] == 1 && !n2.is_dead && n2.op_id == id_relu && n2.inputs[0] == n1.output && use_count[n1.output] == 1) {
      unsigned int dead_mm_tensor = n0.output;
      unsigned int dead_bias_tensor = n1.output;

      n0.op_id = id_fused;
      n0.inputs[2] = n1.inputs[1];
      n0.output = n2.output;

      n1.is_dead = true;
      n2.is_dead = true;
      tensors[dead_mm_tensor].is_dead = true;
      tensors[dead_bias_tensor].is_dead = true;
    }
  }
  for (auto &t : tensors) {
    if (!t.is_persistent && !t.is_dead) {
      t.data = Matrix(t.shape.rows, t.shape.cols, &t_alloc);
      if (t.requires_grad)
        t.grad = Matrix(t.shape.rows, t.shape.cols, &t_alloc);
    }
  }
  compiled_temp_offset = t_alloc.get_offset();
}

void Graph::forward() {
  for (unsigned int i = 0; i != nodes.size(); ++i) {
    if (!nodes[i].is_dead)
      reflect::get_op_meta(nodes[i].op_id).forward(this, i);
  }
}

void Graph::backward() {
  for (auto &t : tensors) {
    if (t.requires_grad && !t.is_dead)
      ops::scale_(t.grad, 0.0f, q, t_alloc.events);
  }
  for (int i = static_cast<int>(nodes.size()) - 1; i != -1; --i) {
    if (!nodes[i].is_dead)
      reflect::get_op_meta(nodes[i].op_id).backward(this, static_cast<unsigned int>(i));
  }
}

void Graph::reset_temp() { t_alloc.reset_to(compiled_temp_offset); }

void Graph::dump() const {
  std::cout << "=== Compiled Graph IR ===\n";
  unsigned int live_idx = 0;
  for (const Node &node : nodes) {
    if (node.is_dead) {
      continue;
    }
    const reflect::OpMetaData &meta = reflect::get_op_meta(node.op_id);
    std::cout << "[" << live_idx++ << "] " << meta.name << "(";
    for (unsigned char j = 0; j < meta.arguments; ++j) {
      unsigned int in_id = node.inputs[j];
      const auto &s = tensors[in_id].shape;
      std::cout << "%" << in_id << "[" << s.rows << "x" << s.cols << "]";
      if (j + 1 < meta.arguments) {
        std::cout << ", ";
      }
    }
    const Shape &out_s = tensors[node.output].shape;
    std::cout << ") -> %" << node.output << "[" << out_s.rows << "x" << out_s.cols << "]\n";
  }
  std::cout << "=========================\n";
}
