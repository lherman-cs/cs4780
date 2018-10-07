cmake_minimum_required(VERSION 3.8)
project(big_dot_2 LANGUAGES C CUDA)
add_executable(big_dot_2 
  main.cu 
  gpu1.cu 
  gpu2.cu
)
