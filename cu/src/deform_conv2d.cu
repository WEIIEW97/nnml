/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils.cuh"
#include <cmath>
#include <algorithm>

__device__ float binterp(const float* M, int C, int H, int W, float x,
                         float y) {
  int x0 = floorf(x);
  int x1 = x0 + 1;
  int y0 = floorf(y);
  int y1 = y0 + 1;

  float wa = (x1 - x) * (y1 - y);
  float wb = (x1 - x) * (y - y0);
  float wc = (x - x0) * (y1 - y);
  float wd = (x - x0) * (y - y0);

  x0 = max(0, min(x0, W - 1));
  x1 = max(0, min(x1, W - 1));
  y0 = max(0, min(y0, H - 1));
  y1 = max(0, min(y1, H - 1));

  float v = 0.0f;
  for (int c = 0; c < C; ++c) {
    float Ia = M[c * H * W + y0 * W + x0];
    float Ib = M[c * H * W + y1 * W + x0];
    float Ic = M[c * H * W + y0 * W + x1];
    float Id = M[c * H * W + y1 * W + x1];
    v += wa * Ia + wb * Ib + wc * Ic + wd * Id;
  }

  return v;
}

__global__ void
bi_interp_kernel(const float* __restrict__ M, // input feature map (C, H, W)
                 const float* __restrict__ x, // x coordinates
                 const float* __restrict__ y, // y coordinates
                 float* __restrict__ out,     // out interpolated values (N, C)
                 int C, int H, int W, int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // get x and y coordinates for this thread
  float fx = x[idx];
  float fy = y[idx];

  for (int c = 0; c < C; ++c) {
    auto v = binterp(M, C, H, W, fx, fy);
    out[idx * C + c] = v;
  }
}

// Host function to perform bilinear interpolation using CUDA
void bi_interp(const float* h_M, // Host input feature map (C, H, W)
               const float* h_x, // Host x coordinates (N)
               const float* h_y, // Host y coordinates (N)
               float* h_output,  // Host output interpolated values (N, C)
               int C, int H, int W, int N) {
  size_t size_M = C * H * W * sizeof(float);
  size_t size_xy = N * sizeof(float);
  size_t size_output = N * C * sizeof(float);

  float* d_M = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_output = nullptr;

  CUDA_CHECK_ERROR(cudaMalloc((void**)&d_M, size_M));
  CUDA_CHECK_ERROR(cudaMalloc((void**)&d_x, size_xy));
  CUDA_CHECK_ERROR(cudaMalloc((void**)&d_y, size_xy));
  CUDA_CHECK_ERROR(cudaMalloc((void**)&d_output, size_output));

  CUDA_CHECK_ERROR(cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_x, h_x, size_xy, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_y, h_y, size_xy, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  bi_interp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_x, d_y, d_output,
                                                       C, H, W, N);

  // Check for kernel launch errors
  CUDA_CHECK_ERROR(cudaGetLastError());

  // Copy the result back to host
  CUDA_CHECK_ERROR(
      cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK_ERROR(cudaFree(d_M));
  CUDA_CHECK_ERROR(cudaFree(d_x));
  CUDA_CHECK_ERROR(cudaFree(d_y));
  CUDA_CHECK_ERROR(cudaFree(d_output));
}

__global__ void deform_conv2d_kernel(const float* __restrict__ M,
                                     const float* __restrict__ weights,
                                     const float* __restrict__ offsets,
                                     float* __restrict__ output, int C_in,
                                     int C_out, int H_in, int W_in, int kH,
                                     int kW, int stride, int padding, int H_out,
                                     int W_out) {
  int out_x = blockIdx.x * blockDim.x + threadIdx.x;
  int out_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (out_x >= W_out || out_y >= H_out)
    return;

  for (int c_out = 0; c_out < C_out; ++c_out) {
    float v = 0.0f;
    int in_y = out_y * stride;
    int in_x = out_x * stride;

    for (int k_y = 0; k_y < kH; ++k_y) {
      for (int k_x = 0; k_x < kW; ++k_x) {
        int offset_idx = 2 * (k_y * kW + k_x);
        float delta_y =
            offsets[offset_idx * H_out * W_out + out_y * W_out + out_x];
        float delta_x =
            offsets[(offset_idx + 1) * H_out * W_out + out_y * W_out + out_x];
        float sample_y = in_y * k_y + delta_y;
        float sample_x = in_x * k_x + delta_x;

        float bi_v = binterp(M, C_in, H_in, W_in, sample_x, sample_y);
        int weight_idx = c_out * C_in * kH * kW + 0 * kH * kW + k_y * kW +
                         k_x; // Assuming C_in=1 for simplicity
        float weight =
            weights[c_out * C_in * kH * kW + 0 * kH * kW + k_y * kW + k_x];

        // Accumulate
        v += weight * bi_v;
      }
    }
    int out_idx = c_out * H_out * W_out + out_y * W_out + out_x;
    output[out_idx] = v;
  }
}