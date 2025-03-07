#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    float tmp = 0.0;
    for (int i = 0; i < K; i++) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = \alpha * (A @ B) + \beta * C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}