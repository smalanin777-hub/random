#ifndef MATRICES_HPP
#define MATRICES_HPP

#include "arena_alloc.hpp"

struct Matrix {
private:
  unsigned int rows;
  unsigned int columns;
  float *data_;
public:

  Matrix(const unsigned int i = 0, const unsigned int j = 0, mem::ArenaAlloc* memory = nullptr);
  Matrix(const Matrix &other_m, mem::ArenaAlloc* memory);
  Matrix &copy(const Matrix &other_m, mem::ArenaAlloc* memory);
  Matrix(const Matrix &other_m) = delete;
  Matrix &operator=(const Matrix &other_m) = delete;
  Matrix(Matrix &&other_m) noexcept;
  Matrix &operator=(Matrix &&other_m) noexcept;
  ~Matrix() = default;
  float *data();
  const float *data() const;
  unsigned int get_rows() const;
  unsigned int get_columns() const;
};

#endif
