#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 512 

float *CPU_big_dot(float *A, float *B, int N);
float *GPU_big_dot(float *A, float *B, int N);
__global__ void multiply(float *a, float *b, float *results, int *N);

/* Helper functions */
float *random(int N);
long long timestamp();

int main(){
  int n = 1 << 20; // 1024 * 1024
  float *a = random(n); 
  float *b = random(n); 

  long long start = timestamp(); 
  float *rcpu = CPU_big_dot(a, b, n);  
  long long end = timestamp();

  float tcpu = (end - start) / 1e6;

  start = timestamp(); 
  float *rgpu = GPU_big_dot(a, b, n);  
  end = timestamp();

  float tgpu = (end - start) / 1e6;

  float error = fabs((*rgpu - *rcpu) / *rcpu) * 100;

  printf("=========== Computed ===========\n");
  printf("     CPU    = %.4f\n", *rcpu);
  printf("     GPU    = %.4f\n", *rgpu);
  printf("     Error  = %.4f%%\n", error);
  printf("================================\n\n");
  
  // Clean up vectors and results
  free(rgpu);
  free(rcpu);
  free(a);
  free(b);

  // Print out metrics
  printf("============ Result ============\n");
  printf("     Tcpu    = %.4fs\n", tcpu);
  printf("     Tgpu    = %.4fs\n", tgpu);
  printf("     Speedup = %.4f\n", tcpu / tgpu);  
  printf("================================\n");

  return 0;
}

float *CPU_big_dot(float *A, float *B, int N){
  float *result = (float *)calloc(1, sizeof(float));
  for(int i = 0; i < N; i++)
    *result += A[i] * B[i];
  return result;
}

float *GPU_big_dot(float *A, float *B, int N){
  // CPU variables 
  int size = sizeof(float) * N;
  float *results = (float *)malloc(size);
  float *result = (float *)calloc(1, sizeof(float));

  // GPU variables 
  int *d_N;
  float *d_A, *d_B, *d_results;

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_results, size);
  cudaMalloc((void **)&d_N, sizeof(int));

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
  
  multiply<<<(N + THREADS_PER_BLOCK -
  1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_A, d_B, d_results, d_N);

  // Copy the results from GPU to CPU
  cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_results);
  cudaFree(d_N);

  // Gather results and sum them up
  for(int i = 0; i < N; i++)
    *result += results[i];
  free(results);

  return result;
}

__global__ void multiply(float *A, float *B, float *results, int *N){
  int index = blockDim.x * blockIdx.x + threadIdx.x; 
  if(index < *N)
    results[index] = A[index] * B[index];
}

float *random(int N){
  float *results = (float *)malloc(sizeof(float) * N);
  for(int i = 0; i < N; i++)
    results[i] = (float)((rand() % 100) + 1);
  return results;
}

long long timestamp(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}
