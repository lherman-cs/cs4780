#include "flags.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

__global__ void kernel(float* output, float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n){
  __shared__ float smem[DIM];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  int tid = threadIdx.x;
  float mul = vector_1[idx] * vector_2[idx];
  __syncthreads();
  smem[tid] = mul;


  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
    if(tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }

  if(tid == 0) output[blockIdx.x] = smem[0];
}

void init_data(float * const ar, int const size){
  for(int i = 0; i < size; i++)
    ar[i] = (float)rand() / (float)RAND_MAX;
}

float gpu_dot_product(unsigned int n){
  int ngpus = 2;
  int i_size = n / ngpus;
  size_t i_bytes = i_size * sizeof(float);
  dim3 block(DIM);
  dim3 grid((i_size + block.x - 1) / block.x);
  int o_size = grid.x;
  size_t o_bytes = o_size * sizeof(float);

  // Allocate array of device memories
  float *d_output[ngpus], *d_vector_1[ngpus], *d_vector_2[ngpus];

  // Allocate array of host memories
  float *h_vector_1[ngpus], *h_vector_2[ngpus], *h_output[ngpus]; 

  // GPU stream
  cudaStream_t stream[ngpus];

  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    
    // Allocate device memories
    cudaMalloc((void **) &d_vector_1[i], i_bytes);
    cudaMalloc((void **) &d_vector_2[i], i_bytes);
    cudaMalloc((void **) &d_output[i], o_bytes);

    // Allocate host memories with page locked
    cudaMallocHost((void **) &h_vector_1[i], i_bytes);
    cudaMallocHost((void **) &h_vector_2[i], i_bytes);
    cudaMallocHost((void **) &h_output[i], o_bytes);

    // Create cuda stream
    cudaStreamCreate(&stream[i]);
  }

  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    init_data(h_vector_1[i], i_size);
    init_data(h_vector_2[i], i_size);
  }

  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    cudaMemcpyAsync(d_vector_1[i], h_vector_1[i], i_bytes, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(d_vector_2[i], h_vector_2[i], i_bytes, cudaMemcpyHostToDevice, stream[i]);

    kernel<<<grid, block, 0, stream[i]>>>(d_output[i], d_vector_1[i], d_vector_2[i],
    i_size);

    cudaMemcpyAsync(h_output[i], d_output[i], o_bytes, cudaMemcpyDeviceToHost, stream[i]);
  }

  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    cudaStreamSynchronize(stream[i]);
  }

  float d_result = 0.0;
  for(int i = 0; i < ngpus; i++){
    for(int j = 0; j < o_size; j++){
      d_result += h_output[i][j];
    }
  }

  float h_result = 0.0;
  for(int i = 0; i < ngpus; i++){
    for(int j = 0; j < i_size; j++){
      h_result += h_vector_1[i][j] * h_vector_2[i][j];
    }
  }

  printf("Host result: %.4f\n", h_result);
  printf("Device result: %.4f\n", d_result);
  printf("Error: %.4f%%\n", (d_result - h_result) / h_result * 100);
  printf("\nNote: Error can occure here because it's relative to the serial version, which is very inacurate\n");
  printf("More information: https://stackoverflow.com/questions/27782731/opencl-reduction-result-wrong-with-large-floats\n");


  for(int i = 0; i < ngpus; i++){
    cudaSetDevice(i);
    cudaFree(d_vector_1[i]);
    cudaFree(d_vector_2[i]);
    cudaFree(d_output[i]);
    
    cudaFreeHost(h_vector_1[i]);
    cudaFreeHost(h_vector_2[i]);
    cudaFreeHost(h_output[i]);

    cudaStreamDestroy(stream[i]);

    cudaDeviceReset();
  }


  return d_result;
}
