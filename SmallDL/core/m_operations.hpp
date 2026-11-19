#ifndef M_OPERATIONS_HPP
#define M_OPERATIONS_HPP

#include "matrices.hpp"

namespace ops {

void sgd_step_(Matrix &data, const Matrix &grad, const float step_size, sycl::queue &q, std::vector<sycl::event> &events);
void scale(const Matrix &m, Matrix &out, float number, sycl::queue &q, std::vector<sycl::event> &events);
void scale_(Matrix &m, float number, sycl::queue &q, std::vector<sycl::event> &events);
void add_scalar_(Matrix &m, float number, sycl::queue &q, std::vector<sycl::event> &events);
void add(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void add_(Matrix &a, const Matrix &b, sycl::queue &q, std::vector<sycl::event> &events);
void add_bias_(Matrix &m, const Matrix &bias, sycl::queue &q, std::vector<sycl::event> &events);
void sum_cols(const Matrix &in, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void matmul_transpose(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void matmul(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void matmul_bias_relu(const Matrix &a, const Matrix &b, const Matrix &bias, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void relu_forward(const Matrix &m, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void relu_backward(const Matrix &m, const Matrix &grad, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void transpose(const Matrix &m, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events);
void mse_loss_forward(const Matrix &pred, const Matrix &target, Matrix &loss_out, sycl::queue &q, std::vector<sycl::event> &events);
void mse_loss_backward(const Matrix &pred, const Matrix &target, Matrix &grad_out, sycl::queue &q, std::vector<sycl::event> &events);
void generate_matrix(Matrix &m, sycl::queue &q, std::vector<sycl::event> &events);

} // namespace ops

#endif
