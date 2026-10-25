#ifndef MATRICES_HPP
#define MATRICES_HPP

#include<sycl/sycl.hpp>

inline sycl::queue global_q{sycl::gpu_selector_v};

template<int N, int K>
struct Power{
  static const int value = N * Power<N, K-1>::value;
};

template<int N>
struct Power<N, 1>{
  static const int value = N;
};

template<int N>
struct Power<N, 0>{
  static const int value = 1;
};

inline const unsigned int THREADSNUM = Power<2, 16>::value;
inline const unsigned int TILE_DIM = Power<2, 4>::value;

struct Matrix{
private:
  unsigned int rows;
  unsigned int columns;
  float * data_;
  mutable sycl::queue q;
public:
  Matrix(const unsigned int i = 0, const unsigned int j = 0, sycl::queue q = global_q);
  Matrix(const Matrix& other_m, sycl::queue q = global_q);
  Matrix& operator=(const Matrix& other_m);
  Matrix(Matrix&& other_m) noexcept;
  Matrix& operator=(Matrix&& other_m) noexcept;
  Matrix& operator*=(const float number);
  Matrix operator*(const float number) const;
  Matrix& operator+=(const float number);
  Matrix operator+(const float number) const;
  Matrix& operator-=(const float number);
  Matrix operator-(const float number) const;
  Matrix& operator*=(const Matrix& other_m);
  Matrix operator*(const Matrix& other_m) const;
  Matrix operator+(const Matrix& other_m) const;
  Matrix& add_bias(const Matrix& bias);
  Matrix relu_forward() const;
  Matrix relu_backward(const Matrix& grad) const;
  Matrix transpose() const;
  float mse_loss_forward(const Matrix& other_m) const;  
  Matrix mse_loss_backward(const Matrix& other_m) const;
  Matrix& generate_matrix();
  float* data();
  const float* data() const;
  unsigned int get_rows() const;
  unsigned int get_columns() const;
  ~Matrix();
};

#endif

