#include "matrices.hpp"
#include <sycl/sycl.hpp>

Matrix::Matrix(const unsigned int i, const unsigned int j, sycl::queue q) : rows(i), columns(j), q(q), last_event() {
  data_ = sycl::malloc_shared<float>(rows * columns, q);
}
Matrix::Matrix(const Matrix &other_m, sycl::queue q) : rows(other_m.rows), columns(other_m.columns), q(q) {
  data_ = sycl::malloc_shared<float>(rows * columns, q);
  last_event = q.memcpy(data_, other_m.data_, rows * columns * sizeof(float));
}
Matrix &Matrix::operator=(const Matrix &other_m) {
  if (this != &other_m) {
    sync();
    sycl::free(data_, q);
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = sycl::malloc_shared<float>(rows * columns, q);
    last_event = q.memcpy(data_, other_m.data_, rows * columns * sizeof(float));
  }
  return *this;
}
Matrix::Matrix(Matrix &&other_m) noexcept : rows(other_m.rows), columns(other_m.columns), data_(other_m.data_), q(other_m.q), last_event(other_m.last_event) {
  other_m.data_ = nullptr;
  other_m.rows = 0;
  other_m.columns = 0;
  other_m.last_event = sycl::event{};
}
Matrix &Matrix::operator=(Matrix &&other_m) noexcept {
  if (this != &other_m) {
    sync();
    sycl::free(data_, q);
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
Matrix::~Matrix() {
  if (data_)
    sync();
    sycl::free(data_, q);
}
