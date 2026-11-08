#include "matrices.hpp"
#include "memory.hpp"
#include <sycl/sycl.hpp>

Matrix::Matrix(const unsigned int i, const unsigned int j, mem::mem_chunk* memory) : rows(i), columns(j), data_((memory) ? memory->alloc_ptr->alloc(i * j) : nullptr), last_event() {
  last_event = memory->q->fill(data_, 0.0f, i * j);
}
Matrix::Matrix(const Matrix &other_m, mem::mem_chunk* memory) : rows(other_m.rows), columns(other_m.columns), data_(memory->alloc_ptr->alloc(rows * columns)) {
  last_event = memory->q->memcpy(data_, other_m.data_, rows * columns * sizeof(float));
}
Matrix &Matrix::copy(const Matrix &other_m, mem::mem_chunk* memory) {
  if (this != &other_m) {
    sync();
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = memory->alloc_ptr->alloc(rows * columns);
    last_event = memory->q->memcpy(data_, other_m.data_, rows * columns * sizeof(float));
  }
  return *this;
}
Matrix::Matrix(Matrix &&other_m) noexcept : rows(other_m.rows), columns(other_m.columns), data_(other_m.data_), last_event(other_m.last_event) {
  other_m.data_ = nullptr;
  other_m.rows = 0;
  other_m.columns = 0;
  other_m.last_event = sycl::event{};
}
Matrix &Matrix::operator=(Matrix &&other_m) noexcept {
  if (this != &other_m) {
    sync();
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = other_m.data_;
    last_event = other_m.last_event;
    other_m.data_ = nullptr;
    other_m.rows = 0;
    other_m.columns = 0;
    other_m.last_event = sycl::event{};
  }
  return *this;
}
void Matrix::sync() { 
  last_event.wait(); 
}
float *Matrix::data() { 
  return data_; 
}
const float *Matrix::data() const { 
  return data_; 
}
unsigned int Matrix::get_rows() const { 
  return rows; 
}
unsigned int Matrix::get_columns() const { 
  return columns; 
}
