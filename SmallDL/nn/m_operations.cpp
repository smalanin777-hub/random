#include "m_operations.hpp"
#include "constants.hpp"
#include <sycl/sycl.hpp>

namespace ops {

void sgd_step_(Matrix &data, const Matrix &grad, const float step_size, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = data.get_rows() * data.get_columns();
  float *d_ptr = data.data();
  const float *g_ptr = grad.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
    for (; i < max; i += stride) {
      d_ptr[i] -= step_size * g_ptr[i];
    }
  }));
}

void scale(const Matrix &m, Matrix &out, const float number, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  const float *d_ptr = m.data();
  float *r_ptr = out.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      r_ptr[i] = d_ptr[i] * number;
    }
  }));
}

void scale_(Matrix &m, const float number, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  float *r_ptr = m.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      r_ptr[i] *= number;
    }
  }));
}

void add_scalar_(Matrix &m, const float number, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  float *r_ptr = m.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned long long stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride) {
      r_ptr[i] += number;
    }
  }));
}

void add(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = a.get_rows() * a.get_columns();
  const float *a_ptr = a.data();
  const float *b_ptr = b.data();
  float *r_ptr = out.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = a_ptr[i] + b_ptr[i];
  }));
}

void add_(Matrix &a, const Matrix &b, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = a.get_rows() * a.get_columns();
  float *a_ptr = a.data();
  const float *b_ptr = b.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned long long i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      a_ptr[i] += b_ptr[i];
  }));
}

void add_bias_(Matrix &m, const Matrix &bias, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  const unsigned int cols = m.get_columns();
  float *r_ptr = m.data();
  const float *bias_ptr = bias.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] += bias_ptr[i % cols];
  }));
}

void sum_cols(const Matrix &in, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned int rows = in.get_rows();
  const unsigned int cols = in.get_columns();
  const float *in_ptr = in.data();
  float *out_ptr = out.data();
  events.push_back(q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> id) {
    unsigned int c = id[0];
    float sum = 0.0f;
    for (unsigned int r = 0; r < rows; ++r) {
      sum += in_ptr[r * cols + c];
    }
    out_ptr[c] = sum;
  }));
}

void matmul_transpose(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned int K = a.get_columns();
  const unsigned int M = b.get_rows();
  const unsigned int N = a.get_rows();
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = a.data();
  const float *data_b = b.data();
  float *data_r = out.data();

  events.push_back(q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));
    h.parallel_for(exec_range, [=](sycl::nd_item<2> item) {
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      float c = 0.0f;
      for (unsigned int i = 0; i < K; i += TILE_DIM) {
        tile_a[l_row * (TILE_DIM + 1) + l_col] = (g_row < N && (i + l_col) < K) ? data_a[g_row * K + (i + l_col)] : 0.0f;
        tile_b[l_row * (TILE_DIM + 1) + l_col] = ((i + l_row) < K && g_col < M) ? data_b[g_col * K + (i + l_row)] : 0.0f;
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
}

void matmul(const Matrix &a, const Matrix &b, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned int K = a.get_columns();
  const unsigned int M = b.get_columns();
  const unsigned int N = a.get_rows();
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = a.data();
  const float *data_b = b.data();
  float *data_r = out.data();
  events.push_back(q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));
    h.parallel_for(exec_range, [=](sycl::nd_item<2> item) {
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      float c = 0.0f;
      for (unsigned int i = 0; i < K; i += TILE_DIM) {
        tile_a[l_row * (TILE_DIM + 1) + l_col] = (g_row < N && (i + l_col) < K) ? data_a[g_row * K + (i + l_col)] : 0.0f;
        tile_b[l_row * (TILE_DIM + 1) + l_col] = ((i + l_row) < K && g_col < M) ? data_b[(i + l_row) * M + g_col] : 0.0f;
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
}

void matmul_bias_relu(const Matrix &a, const Matrix &b, const Matrix &bias, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned int K = a.get_columns();
  const unsigned int M = b.get_columns();
  const unsigned int N = a.get_rows();
  const unsigned int g_rows = ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const float *data_a = a.data();
  const float *data_b = b.data();
  const float *data_bias = bias.data();
  float *data_r = out.data();
  events.push_back(q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::local_accessor<float, 1> tile_b(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    sycl::nd_range exec_range(sycl::range<2>(g_rows, g_cols), sycl::range<2>(TILE_DIM, TILE_DIM));
    h.parallel_for(exec_range, [=](sycl::nd_item<2> item) {
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      float c = 0.0f;
      for (unsigned int i = 0; i < K; i += TILE_DIM) {
        tile_a[l_row * (TILE_DIM + 1) + l_col] = (g_row < N && (i + l_col) < K) ? data_a[g_row * K + (i + l_col)] : 0.0f;
        tile_b[l_row * (TILE_DIM + 1) + l_col] = ((i + l_row) < K && g_col < M) ? data_b[(i + l_row) * M + g_col] : 0.0f;
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (unsigned int k = 0; k < TILE_DIM; ++k)
          c += tile_a[l_row * (TILE_DIM + 1) + k] * tile_b[k * (TILE_DIM + 1) + l_col];
        sycl::group_barrier(item.get_group());
      }
      if (g_row < N && g_col < M) {
        c += data_bias[g_col];
        data_r[g_row * M + g_col] = (c > 0.0f) ? c : 0.0f;
      }
    });
  }));
}

void relu_forward(const Matrix &m, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  const float *d_ptr = m.data();
  float *r_ptr = out.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0.0f) ? d_ptr[i] : 0.0f;
  }));
}

void relu_backward(const Matrix &m, const Matrix &grad, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = m.get_rows() * m.get_columns();
  const float *d_ptr = m.data();
  const float *grad_ptr = grad.data();
  float *r_ptr = out.data();
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      r_ptr[i] = (d_ptr[i] > 0.0f) ? grad_ptr[i] : 0.0f;
  }));
}

void transpose(const Matrix &m, Matrix &out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned int rows = m.get_rows();
  const unsigned int cols = m.get_columns();
  const float *in_ptr = m.data();
  float *out_ptr = out.data();
  const unsigned int g_rows = ((rows + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  const unsigned int g_cols = ((cols + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
  events.push_back(q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile(sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), h);
    h.parallel_for(sycl::nd_range<2>({g_rows, g_cols}, {TILE_DIM, TILE_DIM}), [=](sycl::nd_item<2> item) {
      unsigned int l_row = item.get_local_id(0);
      unsigned int l_col = item.get_local_id(1);
      unsigned int g_row = item.get_global_id(0);
      unsigned int g_col = item.get_global_id(1);
      if (g_row < rows && g_col < cols)
        tile[l_row * (TILE_DIM + 1) + l_col] = in_ptr[g_row * cols + g_col];
      sycl::group_barrier(item.get_group());
      unsigned int out_row = item.get_group(1) * TILE_DIM + l_row;
      unsigned int out_col = item.get_group(0) * TILE_DIM + l_col;
      if (out_row < cols && out_col < rows)
        out_ptr[out_row * rows + out_col] = tile[l_col * (TILE_DIM + 1) + l_row];
    });
  }));
}

void mse_loss_forward(const Matrix &pred, const Matrix &target, Matrix &loss_out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = pred.get_rows() * pred.get_columns();
  const float *pred_ptr = pred.data();
  const float *target_ptr = target.data();
  float *dev_loss = loss_out.data();
  events.push_back(q.single_task([=]() {
    float sum = 0.0f;
    for (unsigned long long i = 0; i < max; ++i) {
      float diff = pred_ptr[i] - target_ptr[i];
      sum += diff * diff;
    }
    dev_loss[0] = sum / static_cast<float>(max);
  }));
}

void mse_loss_backward(const Matrix &pred, const Matrix &target, Matrix &grad_out, sycl::queue &q, std::vector<sycl::event> &events) {
  const unsigned long long max = pred.get_rows() * pred.get_columns();
  const float *pred_ptr = pred.data();
  const float *target_ptr = target.data();
  float *grad_ptr = grad_out.data();
  const float scale_val = 2.0f / static_cast<float>(max);
  events.push_back(q.parallel_for(sycl::range<1>(THREADSNUM), [=](sycl::item<1> item) {
    unsigned int i = item.get_id(0);
    const unsigned int stride = item.get_range(0);
#pragma unroll
    for (; i < max; i += stride)
      grad_ptr[i] = scale_val * (pred_ptr[i] - target_ptr[i]);
  }));
}
void generate_matrix(Matrix &m, sycl::queue &q, std::vector<sycl::event> &events) {
  float *r_ptr = m.data();
  unsigned int max = m.get_rows() * m.get_columns();
  events.push_back(q.submit([&](sycl::handler &h) {
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
}
} // namespace ops
