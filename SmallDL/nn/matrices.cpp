#include "matrices.hpp"
#include "arena_alloc.hpp"

Matrix::Matrix(const unsigned int i, const unsigned int j, mem::ArenaAlloc *memory)
    : rows(i), columns(j), data_((memory) ? memory->alloc(i * j) : nullptr) {}
Matrix::Matrix(const Matrix &other_m, mem::ArenaAlloc *memory)
    : rows(other_m.rows), columns(other_m.columns), data_(memory->alloc(rows * columns)) {
  memory->events.push_back(memory->q.memcpy(data_, other_m.data_, rows * columns * sizeof(float)));
}
Matrix &Matrix::copy(const Matrix &other_m, mem::ArenaAlloc *memory) {
  if (this != &other_m) {
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = memory->alloc(rows * columns);
    memory->events.push_back(memory->q.memcpy(data_, other_m.data_, rows * columns * sizeof(float)));
  }
  return *this;
}
Matrix::Matrix(Matrix &&other_m) noexcept : rows(other_m.rows), columns(other_m.columns), data_(other_m.data_) {
  other_m.data_ = nullptr;
  other_m.rows = 0;
  other_m.columns = 0;
}
Matrix &Matrix::operator=(Matrix &&other_m) noexcept {
  if (this != &other_m) {
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = other_m.data_;
    other_m.data_ = nullptr;
    other_m.rows = 0;
    other_m.columns = 0;
  }
  return *this;
}
float *Matrix::data() { return data_; }
const float *Matrix::data() const { return data_; }
unsigned int Matrix::get_rows() const { return rows; }
unsigned int Matrix::get_columns() const { return columns; }
