#include <stdio.h>
#include <stdlib.h>

float *CPU_big_dot(float *A, float *B, int N);
float *GPU_big_dot(float *A, float *B, int N);
__global__ void multiply(float *a, float *b, float *results);

/* Helper functions */
float *range(float start, float end, float step);
void print_vector(float *ar, int N);

int main(){
  int n = 16;
  float *a = range(1.0, 1.0 + n, 1.0);
  float *b = range(2.0, 2.0 + n, 1.0);
  float *results = GPU_big_dot(a, b, n);  

  print_vector(results, n);

  free(results);
  free(a);
  free(b);
}

float *CPU_big_dot(float *A, float *B, int N){
  float *results = (float *)malloc(sizeof(float) * N);
  for(int i = 0; i < N; i++)
    results[i] = A[i] * B[i];
  return results;
}

float *GPU_big_dot(float *A, float *B, int N){
  // CPU variables 
  int size = sizeof(float) * N;
  float *results = (float *)malloc(size);

  // GPU variables 
  float *d_A, *d_B, *d_results;

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_results, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  
  multiply<<<1,N>>>(d_A, d_B, d_results);

  // Copy the results from GPU to CPU
  cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_results);

  return results;
}

__global__ void multiply(float *A, float *B, float *results){
  results[threadIdx.x] = A[threadIdx.x] * B[threadIdx.x];
}

float *range(float start, float end, float step){
  int n = (int)(end - start) / step + 1;
  float *results = (float *)malloc(sizeof(float) * n);
  float current = start;

  for(int i = 0; i < n; i ++){
    results[i] = current; 
    current += step;
  }
  return results;
}

void print_vector(float *ar, int N){
  for(int i = 0; i < N; i++)
    printf("%f\n", ar[i]);
}
