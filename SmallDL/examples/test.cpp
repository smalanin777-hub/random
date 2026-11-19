#include "layer.hpp"
#include "optimizer.hpp"
#include <chrono>
#include <iostream>

struct MLP {
  Layer l1;
  Layer l2;
  Layer l3;
  Tensor forward(Graph &g, Tensor x) const {
    Tensor a1 = g.emit<ReLU>(l1.forward(g, x));
    Tensor a2 = g.emit<ReLU>(l2.forward(g, a1));
    return l3.forward(g, a2);
  }
};

int main() {
  try {
    const unsigned int BATCH_SIZE = 1000;
    const unsigned int INPUT_DIM = 50;
    const unsigned int HIDDEN_DIM = 500;
    const unsigned int OUTPUT_DIM = 1;
    Graph g;
    Tensor X = g.input(BATCH_SIZE, INPUT_DIM);
    Tensor Y = g.input(BATCH_SIZE, OUTPUT_DIM);
    ops::generate_matrix(g.tensors[X.id].data, g.q, g.g_alloc.events);
    ops::generate_matrix(g.tensors[Y.id].data, g.q, g.g_alloc.events);
    MLP model{make_layer(g, INPUT_DIM, HIDDEN_DIM), make_layer(g, HIDDEN_DIM, HIDDEN_DIM), make_layer(g, HIDDEN_DIM, OUTPUT_DIM)};
    GD optimizer(model, 0.0001f);
    Tensor pred = model.forward(g, X);
    Tensor loss = g.emit<MSELoss>(pred, Y);
    g.dump();
    g.compile();
    g.dump();
    std::cout << "Start\n";
    for (int epoch = 0; epoch <= 1500; ++epoch) {
      auto start = std::chrono::high_resolution_clock::now();
      if (epoch == 30)
        optimizer.set_step(0.0005f);
      if (epoch == 200)
        optimizer.set_step(0.0008f);
      if (epoch == 1100)
        optimizer.set_step(0.001f);
      if (epoch == 1300)
        optimizer.set_step(0.0001f);
      g.forward();
      optimizer.zero_grad(g);
      g.backward();
      optimizer.step(g);
      float host_loss = 0.0f;
      g.q.memcpy(&host_loss, g.tensors[loss.id].data.data(), sizeof(float)).wait();
      g.reset_temp();
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      std::cout << "Epoch " << epoch << " | Loss: " << host_loss << " | Spent time: " << duration.count() << " milli\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
