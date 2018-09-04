#include <stdio.h>
#include <stdlib.h>

float *CPU_big_dot(float *A, float *B, int N);
float *GPU_big_dot(float *A, float *B, int N);

/* Helper functions */
float *range(float start, float end, float step);
void print_vector(float *ar, int N);

int main(){
  int n = 10;
  float *a = range(1.0, 1.0 + n, 1.0);
  float *b = range(2.0, 2.0 + n, 1.0);
  float *results = CPU_big_dot(a, b, n);  

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
