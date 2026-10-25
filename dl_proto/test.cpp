#include"matrices.hpp"
#include"optimizer.hpp"
#include<iostream>
#include<chrono>

int main(){
  const unsigned int BATCH_SIZE = 1000; 
  const unsigned int INPUT_DIM = 50;      
  const unsigned int HIDDEN_DIM = 500;    
  const unsigned int OUTPUT_DIM = 1;    

  std::cout << "Dataset generating (" << BATCH_SIZE << " strings)\n";
    
  Matrix x_mat(BATCH_SIZE, INPUT_DIM);
  Matrix y_mat(BATCH_SIZE, OUTPUT_DIM);
  x_mat.generate_matrix();
  y_mat.generate_matrix();

  auto X = std::make_shared<Tensor>(std::move(x_mat), false);
  auto Y = std::make_shared<Tensor>(std::move(y_mat), false);

  Layer layer1(INPUT_DIM, HIDDEN_DIM);
  Layer layer2(HIDDEN_DIM, HIDDEN_DIM);
  Layer layer3(HIDDEN_DIM, OUTPUT_DIM);

  GD optimizer({layer1.W, layer1.b, layer2.W, layer2.b, layer3.W, layer3.b}, 0.00001f);

  std::cout << "Start\n\n";

  for(int epoch = 0; epoch <= 150; ++epoch) {
    auto start = std::chrono::high_resolution_clock::now();
    if(epoch == 10){
      optimizer.set_step(0.0001f);
    }
    if(epoch == 30){
      optimizer.set_step(0.001f);
    }
    if(epoch == 100){
      optimizer.set_step(0.0001f);
    }
    auto a1 = relu(layer1.forward(X));
    auto a2 = relu(layer2.forward(a1));
    auto prediction = layer3.forward(a2);
    auto loss = mse_loss(prediction, Y);
        
    float current_loss = prediction->data.mse_loss_forward(Y->data);

    optimizer.zero_grad();
    a1->grad *= 0.0f; 
    a2->grad *= 0.0f; 
    prediction->grad *= 0.0f;
        
    loss->backward();
    optimizer.step();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Epoch " << epoch << " | Loss: " << current_loss << " | Spent time: " << duration.count() << " milli\n";
  } 
  return 0;
}

