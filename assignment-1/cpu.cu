#include <stdlib.h>

float *CPU_big_dot(float *A, float *B, int N){
  float *results = (float *)malloc(sizeof(float) * N);
  for(int i = 0; i < N; i++)
    results[i] = A[i] * B[i];
  return results;
}
