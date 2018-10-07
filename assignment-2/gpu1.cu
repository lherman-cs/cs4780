#include "flags.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel1(float* output, float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n){
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

  if(tid == 0) output[blockIdx.x] = smem[0];
}

float gpu_dot_product_1(float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n){
  unsigned int n_blocks = (n + DIM - 1) / DIM;
  float *d_output, *d_vector_1, *d_vector_2, *output;
  unsigned int output_size = n_blocks * sizeof(float),
               input_size = n * sizeof(float);
  float result = 0.0;

  cudaMalloc(&d_output, output_size);
  cudaMalloc(&d_vector_1, input_size);
  cudaMalloc(&d_vector_2, input_size);

  cudaMemcpy(d_vector_1, vector_1, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector_2, input_size, cudaMemcpyHostToDevice);

  kernel1<<<n_blocks, DIM>>>(d_output, d_vector_1, d_vector_2, n);
  
  output = (float*)malloc(output_size);
  cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_vector_1);
  cudaFree(d_vector_2);

  for(unsigned int i = 0; i < n_blocks; i++) result += output[i];
  free(output);
  return result;
}