#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <sycl/sycl.hpp>
#include "memory.hpp"
#include "constants.hpp"

struct Matrix {
private:
  unsigned int rows;
  unsigned int columns;
  float *data_;
public:
  sycl::event last_event;

  Matrix(const unsigned int i = 0, const unsigned int j = 0, mem::mem_chunk* memory = nullptr);
  Matrix(const Matrix &other_m, mem::mem_chunk* memory);
  Matrix &copy(const Matrix &other_m, mem::mem_chunk* memory);
  Matrix(const Matrix &other_m) = delete;
  Matrix &operator=(const Matrix &other_m) = delete;
  Matrix(Matrix &&other_m) noexcept;
  Matrix &operator=(Matrix &&other_m) noexcept;
  ~Matrix() = default;
  void sync();
  float *data();
  const float *data() const;
  unsigned int get_rows() const;
  unsigned int get_columns() const;
};

#endif
