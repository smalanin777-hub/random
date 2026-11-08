#include "matrices.hpp"
#include "m_operations.hpp"
#include "optimizer.hpp"
#include "constants.hpp"
#include <chrono>
#include <iostream>

int main() {
  const unsigned int BATCH_SIZE = 1000;
  const unsigned int INPUT_DIM = 50;
  const unsigned int HIDDEN_DIM = 500;
  const unsigned int OUTPUT_DIM = 1;

  std::cout << "Dataset generating (" << BATCH_SIZE << " strings)\n";
  Matrix x_mat(BATCH_SIZE, INPUT_DIM, &global_chunk);
  Matrix y_mat(BATCH_SIZE, OUTPUT_DIM, &global_chunk);
  ops::generate_matrix(x_mat, &global_chunk);
  ops::generate_matrix(y_mat, &global_chunk);

  Matrix x_copy(x_mat, &global_chunk);
  Matrix y_copy(y_mat, &global_chunk);

  auto X = std::make_shared<Tensor>(std::move(x_copy), false, &temp_chunk);
  auto Y = std::make_shared<Tensor>(std::move(y_copy), false, &temp_chunk);

  Layer layer1(INPUT_DIM, HIDDEN_DIM);
  Layer layer2(HIDDEN_DIM, HIDDEN_DIM);
  Layer layer3(HIDDEN_DIM, OUTPUT_DIM);

  GD optimizer({layer1.W, layer1.b, layer2.W, layer2.b, layer3.W, layer3.b}, 0.0001f);

  std::cout << "Start\n";
  for (int epoch = 0; epoch <= 1500; ++epoch) {
    auto start = std::chrono::high_resolution_clock::now();

    if (epoch == 30)  optimizer.set_step(0.0005f);
    if (epoch == 200)  optimizer.set_step(0.0008f);
    if (epoch == 1100)  optimizer.set_step(0.001f);
    if (epoch == 1300) optimizer.set_step(0.0001f);

    auto a1 = relu(layer1.forward(X));
    auto a2 = relu(layer2.forward(a1));
    auto prediction = layer3.forward(a2);
    auto loss = mse_loss(prediction, Y);

    float current_loss = ops::mse_loss_forward(prediction->data, Y->data, &temp_chunk);

    optimizer.zero_grad();
    loss->backward();
    optimizer.step();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Epoch " << epoch << " | Loss: " << current_loss << " | Spent time: " << duration.count() << " milli\n";

    temp_alloc.reset();
  }
  return 0;
}
