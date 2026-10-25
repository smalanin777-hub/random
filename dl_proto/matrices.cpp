#include"matrices.hpp"
#include<sycl/sycl.hpp>
#include<iostream>
#include<random>

Matrix::Matrix(const unsigned int i, const unsigned int j, sycl::queue q) : rows(i), columns(j), q(q) { 
  data_ = sycl::malloc_shared<float>(rows * columns, q);
}
Matrix::Matrix(const Matrix& other_m, sycl::queue q) : rows(other_m.rows), columns(other_m.columns), q(q) {
  data_ = sycl::malloc_shared<float>(rows * columns, q); 
  q.memcpy(data_, other_m.data_, rows * columns * sizeof(float)).wait();
}
Matrix& Matrix::operator=(const Matrix& other_m){
  if(this != &other_m){
    sycl::free(data_, q);
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = sycl::malloc_shared<float>(rows * columns, q);
    q.memcpy(data_, other_m.data_, rows * columns * sizeof(float)).wait();
  }
  return *this;
}
Matrix::Matrix(Matrix&& other_m) noexcept : rows(other_m.rows), columns(other_m.columns), data_(other_m.data_), q(other_m.q) {
  other_m.data_ = nullptr;
  other_m.rows = 0;
  other_m.columns = 0;
}
Matrix& Matrix::operator=(Matrix&& other_m) noexcept{
  if(this != &other_m){
    sycl::free(data_, q);
    rows = other_m.rows;
    columns = other_m.columns;
    data_ = other_m.data_;
    other_m.data_ = nullptr;
    other_m.rows = 0;
    other_m.columns = 0;
  }
  return *this;
}
Matrix& Matrix::operator*=(const float number){
  const unsigned long long max = this->rows * this->columns;
  float *r_ptr = this->data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0); 
    #pragma unroll
    for(; i < max; i += stride){
      const float prev_val = r_ptr[i];
      r_ptr[i] = prev_val * number;
    }
  }).wait();
  return *this;
}
Matrix Matrix::operator*(const float number) const{
  Matrix res(rows, columns, q);
  const unsigned long long max = this->rows * this->columns;
  const float *d_ptr = this->data_;
  float *r_ptr = res.data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride){
      r_ptr[i] = d_ptr[i] * number; 
    }
  }).wait();
  return res;
}
Matrix& Matrix::operator+=(const float number){
  const unsigned long long max = this->rows * this->columns;
  float *r_ptr = this->data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0); 
    #pragma unroll
    for(; i < max; i += stride){
      const float prev_val = r_ptr[i];
      r_ptr[i] = prev_val + number;
    }
  }).wait();
  return *this;
}
Matrix Matrix::operator+(const float number) const{
  Matrix res(rows, columns, q);
  const unsigned long long max = this->rows * this->columns;
  const float *d_ptr = this->data_;
  float *r_ptr = res.data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride){
      r_ptr[i] = d_ptr[i] + number; 
    }
  }).wait();
  return res;
}
Matrix& Matrix::operator-=(const float number){
  const unsigned long long max = this->rows * this->columns;
  float *r_ptr = this->data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0); 
    #pragma unroll
    for(; i < max; i += stride){
      const float prev_val = r_ptr[i];
      r_ptr[i] = prev_val - number;
    }
  }).wait();
  return *this;
}
Matrix Matrix::operator-(const float number) const{
  Matrix res(rows, columns, q);
  const unsigned long long max = this->rows * this->columns;
  const float *d_ptr = this->data_;
  float *r_ptr = res.data_;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride){
      r_ptr[i] = d_ptr[i] - number; 
    }
  }).wait();
  return res;
}
Matrix& Matrix::operator*=(const Matrix& other_m){
  if(!this->data() || !other_m.data() || (this->columns != other_m.rows)){
    std::cout << "\nUnavaliable operations!\n";
    abort();
  }
  const unsigned int K = this->columns;
  const unsigned int M = other_m.columns;
  const unsigned int N = this->rows; 
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = this->data();
  const float *data_b = other_m.data();
  Matrix r(N, M, q);
  float *data_r = r.data();
  q.submit([&](sycl::handler& h){
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));
    
    h.parallel_for(exec_range, [=](sycl::nd_item<2> item){
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      
      float c = 0;
   
      for(unsigned int i = 0; i < K; i += TILE_DIM){
        if(g_row < N && (i + l_col) < K)
          tile_a[l_row * (TILE_DIM + 1) + l_col] = data_a[g_row * K + (i + l_col)];
        else
          tile_a[l_row * (TILE_DIM + 1) + l_col] = 0;
        if((i + l_row) < K && g_col < M)
          tile_b[l_row * (TILE_DIM + 1) + l_col] = data_b[(i + l_row) * M + g_col];
        else 
          tile_b[l_row * (TILE_DIM + 1) + l_col] = 0;
        sycl::group_barrier(item.get_group());
        #pragma unroll 
        for(unsigned int k = 0; k < TILE_DIM; ++k)
          c += tile_a[l_row * (TILE_DIM + 1) + k] * tile_b[k * (TILE_DIM + 1) + l_col]; 
        sycl::group_barrier(item.get_group());
      }
      if(g_row < N && g_col < M)
        data_r[g_row * M + g_col] = c;
    });
  }).wait(); 
  sycl::free(data_, q);
  rows = r.rows;
  columns = r.columns;
  data_ = r.data_;
  r.data_ = nullptr;
  r.rows = 0;
  r.columns = 0;
  return *this;
}
Matrix Matrix::operator*(const Matrix& other_m) const{
  if(!this->data() || !other_m.data() || (this->columns != other_m.rows)){
    std::cout << "\nUnavaliable operations!\n";
    abort();
  }
  const unsigned int K = this->columns;
  const unsigned int M = other_m.columns;
  const unsigned int N = this->rows; 
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = this->data();
  const float *data_b = other_m.data();
  Matrix r(N, M, q);
  float *data_r = r.data();
  q.submit([&](sycl::handler& h){
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));
    
    h.parallel_for(exec_range, [=](sycl::nd_item<2> item){
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      
      float c = 0;
   
      for(unsigned int i = 0; i < K; i += TILE_DIM){
        if(g_row < N && (i + l_col) < K)
          tile_a[l_row * (TILE_DIM + 1) + l_col] = data_a[g_row * K + (i + l_col)];
        else
          tile_a[l_row * (TILE_DIM + 1) + l_col] = 0;
        if((i + l_row) < K && g_col < M)
          tile_b[l_row * (TILE_DIM + 1) + l_col] = data_b[(i + l_row) * M + g_col];
        else 
          tile_b[l_row * (TILE_DIM + 1) + l_col] = 0;
        sycl::group_barrier(item.get_group());
        #pragma unroll 
        for(unsigned int k = 0; k < TILE_DIM; ++k)
          c += tile_a[l_row * (TILE_DIM + 1) + k] * tile_b[k * (TILE_DIM + 1) + l_col]; 
        sycl::group_barrier(item.get_group());
      }
      if(g_row < N && g_col < M)
        data_r[g_row * M + g_col] = c;
    });
  }).wait(); 
  return r; 
}
Matrix Matrix::operator+(const Matrix& other_m) const{
  if(rows != other_m.rows || columns != other_m.columns){
    std::cout << "Unavailable operation\n";
    abort();
  }
  Matrix r(rows, columns, q);
  const unsigned long long max = rows * columns;
  const float* a_ptr = this->data();
  const float* b_ptr = other_m.data();
  float* r_ptr = r.data();
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned long long i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride)
      r_ptr[i] = a_ptr[i] + b_ptr[i];
  }).wait();
  return r;
}
Matrix& Matrix::add_bias(const Matrix& bias){
  if(bias.get_rows() != 1 || bias.get_columns() != columns){
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = rows * columns;
  const unsigned int cols = columns;
  float* r_ptr = this->data();
  const float* bias_ptr = bias.data();
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride)
      r_ptr[i] += bias_ptr[i % cols];
  }).wait();
  return *this;
}
Matrix Matrix::relu_forward() const{
  Matrix r(rows, columns, q);
  const unsigned long long max = rows * columns;
  const float* d_ptr = this->data();
  float* r_ptr = r.data();
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0) ? d_ptr[i] : 0;  
  }).wait();
  return r;
}
Matrix Matrix::relu_backward(const Matrix& grad) const{
  if(grad.get_rows() != rows || grad.get_columns() != columns){
    std::cout << "Unavailable operation\n";
    abort();
  }
  Matrix r(rows, columns, q);
  const unsigned long long max = rows * columns;
  const float* d_ptr = this->data();
  const float* grad_ptr = grad.data();
  float* r_ptr = r.data();
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0) ? grad_ptr[i] : 0; 
  }).wait();
  return r;
}
Matrix Matrix::transpose() const{
  Matrix res(columns, rows, q);
  const unsigned long long r_max = rows;
  const unsigned long long c_max = columns;
  float* res_ptr = res.data_;
  const float* data_ptr = this->data();
  q.parallel_for(sycl::range<2>{TILE_DIM, TILE_DIM}, [=](sycl::item<2> item){
    unsigned long long row = item.get_id(0);
    unsigned long long column = item.get_id(1);
    const unsigned int r_stride = item.get_range(0);
    const unsigned int c_stride = item.get_range(1);
    #pragma unroll
    for(; row < r_max; row += r_stride){
      #pragma unroll 
      for(unsigned int c_iter = column; c_iter < c_max; c_iter += c_stride){
        res_ptr[c_iter * r_max + row] = data_ptr[row * c_max + c_iter];
      }
    }
  }).wait();  
  return res;
}
float Matrix::mse_loss_forward(const Matrix& other_m) const{
  if(other_m.get_rows() != rows || other_m.get_columns() != columns){
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = rows * columns;
  const float* pred_ptr = this->data();
  const float* target_ptr = other_m.data();
  float* loss = sycl::malloc_shared<float>(1, q); 
  loss[0] = 0.0f;
  q.submit([&](sycl::handler& h){
    auto red = sycl::reduction(loss, sycl::plus<>());
    h.parallel_for(sycl::range<1>(THREADSNUM), red, [=](sycl::item<1> item, auto& acc){
      unsigned int i = item.get_id(0);
      const unsigned int stride = item.get_range(0);
      float loc_sum = 0.0f;
      #pragma unroll
      for(; i < max; i += stride){
        float diff = pred_ptr[i] - target_ptr[i];
        loc_sum += diff * diff;
      }
      acc.combine(loc_sum);
    });
  }).wait();
  float f_loss = loss[0] / static_cast<float>(max);
  sycl::free(loss, q);
  return f_loss;
}
Matrix Matrix::mse_loss_backward(const Matrix& other_m) const{
  if(other_m.get_rows() != rows || other_m.get_columns() != columns){
    std::cout << "Unavailable operation\n";
    abort();
  }
  Matrix res(rows, columns, q);
  const unsigned long long max = rows * columns;
  const float* pred_ptr = this->data();
  const float* target_ptr = other_m.data();
  float* grad_ptr = res.data();
  const float scale = 2.0f / max;
  q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item){
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
    #pragma unroll
    for(; i < max; i += stride)
      grad_ptr[i] = scale * (pred_ptr[i] - target_ptr[i]); 
  }).wait();
  return res;
}
Matrix& Matrix::generate_matrix(){
  float* r_ptr = data_;
  unsigned int max = rows * columns;
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
  for(unsigned int i = 0; i < max; ++i) {
    r_ptr[i] = dis(gen);
  }
  return *this; 
}
float* Matrix::data(){
  return data_;
}
const float* Matrix::data() const{
  return data_;
}
unsigned int Matrix::get_rows() const{
  return this->rows;
}
unsigned int Matrix::get_columns() const{
  return this->columns;
}
Matrix::~Matrix(){
  if (data_)
    sycl::free(data_, q);
}
