#ifndef M_OPERATIONS_HPP
#define M_OPERATIONS_HPP
#include"matrices.hpp"

namespace ops{

  Matrix scale(const Matrix &m, float number);
  void scale_(Matrix &m, float number);
  void add_scalar_(Matrix &m, float number);
  Matrix add(const Matrix &a, const Matrix &b);
  void add_(Matrix &a, const Matrix &b);
  void add_bias_(Matrix &m, const Matrix &bias);
  Matrix matmul(const Matrix &a, const Matrix &b);
  Matrix relu_forward(const Matrix &m);
  Matrix relu_backward(const Matrix &m, const Matrix &grad);
  Matrix transpose(const Matrix &m);
  float mse_loss_forward(Matrix &pred, const Matrix &target);
  Matrix mse_loss_backward(Matrix &pred, const Matrix &target); 
  void generate_matrix(Matrix &m);

}

#endif
