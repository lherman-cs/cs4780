#include "flags.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel2(float* result, float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n){
  __shared__ float smem[DIM];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  // printf("%u\n", idx);
  int tid = threadIdx.x;
  smem[tid] = vector_1[idx] * vector_2[idx];
  __syncthreads();


  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
    if(tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }

  if(tid == 0) atomicAdd(result, smem[0]);
}

float gpu_dot_product_2(float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n){
  unsigned int n_blocks = (n + DIM - 1) / DIM;
  float *d_result, *d_vector_1, *d_vector_2, *result;
  unsigned int output_size = sizeof(float),
               input_size = n * sizeof(float);

  cudaMalloc(&d_result, output_size);
  cudaMalloc(&d_vector_1, input_size);
  cudaMalloc(&d_vector_2, input_size);

  cudaMemcpy(d_vector_1, vector_1, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector_2, input_size, cudaMemcpyHostToDevice);

  kernel2<<<n_blocks, DIM>>>(d_result, d_vector_1, d_vector_2, n);
  
  result = (float*)malloc(output_size);
  cudaMemcpy(result, d_result, output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_result);
  cudaFree(d_vector_1);
  cudaFree(d_vector_2);

  return *result;
}