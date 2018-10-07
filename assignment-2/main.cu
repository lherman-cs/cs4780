#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "flags.h"

typedef float (*dot_f)(float* __restrict__, float* __restrict__, unsigned int);
extern float gpu_dot_product_1(float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n);
extern float gpu_dot_product_2(float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n);

float bench(dot_f f, float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n, const char *name);
float *range(unsigned int n);
void print_error(float expected, float actual);


int main(){
  float *a = range(N);
  float *b = range(N);

  float t1 = bench(gpu_dot_product_1, a, b, N, "kernel1");
  float t2 = bench(gpu_dot_product_2, a, b, N, "kernel2");

  free(a);
  free(b);
}

float bench(dot_f f, float* __restrict__ vector_1, float* __restrict__ vector_2, unsigned int n, const char* name){
  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  float result = f(vector_1, vector_2, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  printf("================== %s ==================\n", name);
  printf("          Result : %f\n", result);
  printf("          Elapsed: %fms\n\n", elapsed);

  return elapsed;
}

float *range(unsigned int n){
  float *results = (float *)malloc(sizeof(float) * n);

  for(int i = 0; i < n; i ++) results[i] = (float)rand()/RAND_MAX;
  return results;
}

void print_error(float expected, float actual){
  float error = fabs((actual - expected) / expected) * 100;
  printf("Error: %.4f%%\n", error);
}