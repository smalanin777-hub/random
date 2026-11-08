#ifndef M_OPERATIONS_HPP
#define M_OPERATIONS_HPP

#include"matrices.hpp"
#include"memory.hpp"

namespace ops{

  Matrix scale(const Matrix &m, float number, mem::mem_chunk* memory);
  void scale_(Matrix &m, float number, mem::mem_chunk* memory);
  void add_scalar_(Matrix &m, float number, mem::mem_chunk* memory);
  Matrix add(const Matrix &a, const Matrix &b, mem::mem_chunk* memory);
  void add_(Matrix &a, const Matrix &b, mem::mem_chunk* memory);
  void add_bias_(Matrix &m, const Matrix &bias, mem::mem_chunk* memory);
  Matrix matmul(const Matrix &a, const Matrix &b, mem::mem_chunk* memory);
  Matrix relu_forward(const Matrix &m, mem::mem_chunk* memory);
  Matrix relu_backward(const Matrix &m, const Matrix &grad, mem::mem_chunk* memory);
  Matrix transpose(const Matrix &m, mem::mem_chunk* memory);
  float mse_loss_forward(Matrix &pred, const Matrix &target, mem::mem_chunk* memory);
  Matrix mse_loss_backward(Matrix &pred, const Matrix &target, mem::mem_chunk* memory); 
  void generate_matrix(Matrix &m, mem::mem_chunk* memory);

}

#endif
