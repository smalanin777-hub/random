#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <sycl/sycl.hpp>
#include "constants.hpp"

struct Matrix {
private:
  unsigned int rows;
  unsigned int columns;
  float *data_;
public:
  sycl::queue q;
  sycl::event last_event;

  Matrix(const unsigned int i = 0, const unsigned int j = 0, sycl::queue q = global_q);
  Matrix(const Matrix &other_m, sycl::queue q = global_q);
  Matrix &operator=(const Matrix &other_m);
  Matrix(Matrix &&other_m) noexcept;
  Matrix &operator=(Matrix &&other_m) noexcept;
  ~Matrix();
  void sync();
  float *data();
  const float *data() const;
  unsigned int get_rows() const;
  unsigned int get_columns() const;
};

#endif
