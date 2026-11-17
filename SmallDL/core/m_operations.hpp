#ifndef M_OPERATIONS_HPP
#define M_OPERATIONS_HPP

#include"matrices.hpp"

namespace ops{

  Matrix scale(const Matrix &m, float number, mem::ArenaAlloc* memory);
  void scale_(Matrix &m, float number, mem::ArenaAlloc* memory);
  void add_scalar_(Matrix &m, float number, mem::ArenaAlloc* memory);
  Matrix add(const Matrix &a, const Matrix &b, mem::ArenaAlloc* memory);
  void add_(Matrix &a, const Matrix &b, mem::ArenaAlloc* memory);
  void add_bias_(Matrix &m, const Matrix &bias, mem::ArenaAlloc* memory);
  Matrix matmul(const Matrix &a, const Matrix &b, mem::ArenaAlloc* memory);
  Matrix relu_forward(const Matrix &m, mem::ArenaAlloc* memory);
  Matrix relu_backward(const Matrix &m, const Matrix &grad, mem::ArenaAlloc* memory);
  Matrix transpose(const Matrix &m, mem::ArenaAlloc* memory);
  float mse_loss_forward(Matrix &pred, const Matrix &target, mem::ArenaAlloc* memory);
  Matrix mse_loss_backward(Matrix &pred, const Matrix &target, mem::ArenaAlloc* memory); 
  void generate_matrix(Matrix &m, mem::ArenaAlloc* memory);

}

#endif
