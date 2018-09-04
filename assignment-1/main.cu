#include <stdio.h>
#include <stdlib.h>

float *CPU_big_dot(float *A, float *B, int N);
float *GPU_big_dot(float *A, float *B, int N);

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

int main(){
  float *a = range(1.0, 10.0, 1.0);
  float *b = range(2.0, 11.0, 1.0);
  float *results = CPU_big_dot(a, b, 10);  

  printf("%f", 

  free(a);
  free(b);
}
