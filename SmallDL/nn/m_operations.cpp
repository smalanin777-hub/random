#include "m_operations.hpp"
#include "constants.hpp"
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

namespace ops {

Matrix scale(const Matrix &m, const float number, mem::ArenaAlloc *memory) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  Matrix r(m.get_rows(), m.get_columns(), memory);
  const float *d_ptr = m.data();
  float *r_ptr = r.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      r_ptr[i] = d_ptr[i] * number;
    }
  }));
  return std::move(r);
}
void scale_(Matrix &m, const float number, mem::ArenaAlloc *memory) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  float *r_ptr = m.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      const float prev_val = r_ptr[i];
      r_ptr[i] = prev_val * number;
    }
  }));
  return;
}
void add_scalar_(Matrix &m, const float number, mem::ArenaAlloc *memory) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  float *r_ptr = m.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      const float prev_val = r_ptr[i];
      r_ptr[i] = prev_val + number;
    }
  }));
  return;
}
Matrix add(const Matrix &a, const Matrix &b, mem::ArenaAlloc *memory) {
  if (a.get_rows() != b.get_rows() || a.get_columns() != b.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = a.get_rows() * a.get_columns();
  Matrix r(a.get_rows(), a.get_columns(), memory);
  const float *a_ptr = a.data();
  const float *b_ptr = b.data();
  float *r_ptr = r.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = a_ptr[i] + b_ptr[i];
  }));
  return std::move(r);
}
void add_(Matrix &a, const Matrix &b, mem::ArenaAlloc *memory) {
  if (a.get_rows() != b.get_rows() || a.get_columns() != b.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = a.get_rows() * a.get_columns();
  float *a_ptr = a.data();
  const float *b_ptr = b.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      a_ptr[i] += b_ptr[i];
  }));
  return;
}
void add_bias_(Matrix &m, const Matrix &bias, mem::ArenaAlloc *memory) {
  if (bias.get_rows() != 1 || bias.get_columns() != m.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = m.get_rows() * m.get_columns();
  const unsigned int cols = m.get_columns();
  float *r_ptr = m.data();
  const float *bias_ptr = bias.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] += bias_ptr[i % cols];
  }));
  return;
}
Matrix matmul(const Matrix &a, const Matrix &b, mem::ArenaAlloc *memory) {
  if (!a.data() || !b.data() || (a.get_columns() != b.get_rows())) {
    std::cout << "\nUnavaliable operations!\n";
    abort();
  }
  const unsigned int K = a.get_columns();
  const unsigned int M = b.get_columns();
  const unsigned int N = a.get_rows();
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = a.data();
  const float *data_b = b.data();
  Matrix r(N, M, memory);
  float *data_r = r.data();
  memory->events.push_back(memory->q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));

    h.parallel_for(exec_range, [=](sycl::nd_item<2> item) {
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      float c = 0;
      for (unsigned int i = 0; i < K; i += TILE_DIM) {
        if (g_row < N && (i + l_col) < K)
          tile_a[l_row * (TILE_DIM + 1) + l_col] = data_a[g_row * K + (i + l_col)];
        else
          tile_a[l_row * (TILE_DIM + 1) + l_col] = 0;
        if ((i + l_row) < K && g_col < M)
          tile_b[l_row * (TILE_DIM + 1) + l_col] = data_b[(i + l_row) * M + g_col];
        else
          tile_b[l_row * (TILE_DIM + 1) + l_col] = 0;
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (unsigned int k = 0; k < TILE_DIM; ++k)
          c += tile_a[l_row * (TILE_DIM + 1) + k] * tile_b[k * (TILE_DIM + 1) + l_col];
        sycl::group_barrier(item.get_group());
      }
      if (g_row < N && g_col < M)
        data_r[g_row * M + g_col] = c;
    });
  }));
  return std::move(r);
}
Matrix relu_forward(const Matrix &m, mem::ArenaAlloc *memory) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  Matrix r(m.get_rows(), m.get_columns(), memory);
  const float *d_ptr = m.data();
  float *r_ptr = r.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0) ? d_ptr[i] : 0;
  }));
  return std::move(r);
}
Matrix relu_backward(const Matrix &m, const Matrix &grad, mem::ArenaAlloc *memory) {
  if (grad.get_rows() != m.get_rows() || grad.get_columns() != m.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = m.get_rows() * m.get_columns();
  Matrix r(m.get_rows(), m.get_columns(), memory);
  const float *d_ptr = m.data();
  const float *grad_ptr = grad.data();
  float *r_ptr = r.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0) ? grad_ptr[i] : 0;
  }));
  return std::move(r);
}
Matrix transpose(const Matrix &m, mem::ArenaAlloc *memory) {
  Matrix r(m.get_columns(), m.get_rows(), memory);
  const unsigned long long r_max = m.get_rows();
  const unsigned long long c_max = m.get_columns();
  float *r_ptr = r.data();
  const float *data_ptr = m.data();
  memory->events.push_back(memory->q.parallel_for(sycl::range<2>{TILE_DIM, TILE_DIM}, [=](sycl::item<2> item) {
    unsigned long long row = item.get_id(0);
    unsigned long long column = item.get_id(1);
    const unsigned int r_stride = item.get_range(0);
    const unsigned int c_stride = item.get_range(1);
#pragma unroll
    for (; row < r_max; row += r_stride) {
#pragma unroll
      for (unsigned int c_iter = column; c_iter < c_max; c_iter += c_stride) {
        r_ptr[c_iter * r_max + row] = data_ptr[row * c_max + c_iter];
      }
    }
  }));
  return std::move(r);
}
float mse_loss_forward(Matrix &pred, const Matrix &target, mem::ArenaAlloc *memory) {
  if (pred.get_rows() != target.get_rows() || pred.get_columns() != target.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = pred.get_rows() * pred.get_columns();
  const float *pred_ptr = pred.data();
  const float *target_ptr = target.data();
  float *dev_loss = memory->alloc(1);
  memory->q.fill(dev_loss, 0.0f, 1).wait();
  memory->q
      .submit([&](sycl::handler &h) {
        auto red = sycl::reduction(dev_loss, sycl::plus<>());
        h.parallel_for(sycl::range<1>(THREADSNUM), red, [=](sycl::item<1> item, auto &acc) {
          unsigned int i = item.get_id(0);
          const unsigned int stride = item.get_range(0);
          float loc_sum = 0.0f;
#pragma unroll
          for (; i < max; i += stride) {
            float diff = pred_ptr[i] - target_ptr[i];
            loc_sum += diff * diff;
          }
          acc.combine(loc_sum);
        });
      })
      .wait();
  float host_loss = 0.0f;
  memory->q.memcpy(&host_loss, dev_loss, sizeof(float)).wait();
  return host_loss / static_cast<float>(max);
}
Matrix mse_loss_backward(Matrix &pred, const Matrix &target, mem::ArenaAlloc *memory) {
  if (pred.get_rows() != target.get_rows() || pred.get_columns() != target.get_columns()) {
    std::cout << "Unavailable operation\n";
    abort();
  }
  const unsigned long long max = pred.get_rows() * pred.get_columns();
  Matrix r(pred.get_rows(), pred.get_columns(), memory);
  const float *pred_ptr = pred.data();
  const float *target_ptr = target.data();
  float *grad_ptr = r.data();
  const float scale = 2.0f / max;
  memory->events.push_back(memory->q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      grad_ptr[i] = scale * (pred_ptr[i] - target_ptr[i]);
  }));
  return std::move(r);
}
void generate_matrix(Matrix &m, mem::ArenaAlloc *memory) {
  float *r_ptr = m.data();
  unsigned int max = m.get_rows() * m.get_columns();
  memory->events.push_back(memory->q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(max), [=](sycl::id<1> id) {
      unsigned int i = id[0];
      unsigned int seed = i * 2654435761u;
      seed ^= seed >> 16;
      seed *= 0x85ebca6b;
      seed ^= seed >> 13;
      seed *= 0xc2b2ae35;
      seed ^= seed >> 16;
      r_ptr[i] = (static_cast<float>(seed) / 4294967295.0f) - 0.5f;
    });
  }));
  return;
}
} // namespace ops
