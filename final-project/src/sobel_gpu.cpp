#include <stdio.h>
#include <string.h>
#include <vector>
#include "sobel.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char *kernel = 
"void __kernel find_edge(global read_only char *in, global write_only char *out,\n"
"const unsigned int w, const unsigned int h) {\n"
"size_t y = get_global_id(0);\n"
"size_t x = get_global_id(1);\n"
"size_t id = y * x;\n"
"// Compute gradient in +ve x direction\n"
"float gradient_X = in[ (x-1) + (y-1) * w ]\n"
"- in[ (x+1) + (y-1) * w ]\n"
"+ 2 * in[ (x-1) +  y    * w ]\n"
"- 2 * in[ (x+1) +  y    * w ]\n"
"+ in[ (x-1) + (y+1) * w ]\n"
"- in[ (x+1) + (y+1) * w ];\n"
"// Compute gradient in +ve y direction\n"
"float gradient_Y = in[ (x-1) + (y-1) * w ]\n"
"+ 2 * in[  x    + (y-1) * w ]\n"
"+ in[ (x+1) + (y-1) * w ]\n"
"- in[ (x-1) + (y+1) * w ]\n"
"- 2 * in[  x    + (y+1) * w ]\n"
"- in[ (x+1) + (y+1) * w ];\n"
"int value = ceil(sqrt(pow(gradient_X, 2) + pow(gradient_Y, 2)));\n"
"out[id] = 255 - value;\n"
"}\n";

void handle(cl_int err) {
  if (err == CL_SUCCESS) return;
  std::__throw_runtime_error(std::to_string(err).data());
}

cl_int get_platform(const char *vendor, cl_platform_id *chosen) {
  cl_uint num_platforms;
  cl_int ret;

  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if(ret != CL_SUCCESS) return ret;
  
  cl_platform_id *platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));

  ret = clGetPlatformIDs(num_platforms, platforms, NULL);
  if(ret != CL_SUCCESS) {
    free(platforms);
    return ret;
  }

  size_t size_ret;
  char *_vendor = NULL;
  for(cl_uint i = 0; i < num_platforms; i++){
    ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &size_ret);
    if(ret != CL_SUCCESS) break;

    _vendor = (char*)realloc((void*)_vendor, size_ret);
    ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, size_ret, _vendor, NULL);
    if(ret != CL_SUCCESS) break;

    printf("Found %s!\n", _vendor);
    if(strstr(_vendor, vendor) != NULL) {
      *chosen = platforms[i];
      ret = CL_SUCCESS;
      break;
    }
  }
  
  free(_vendor);
  free(platforms);
  return ret;
}

cl_int init(cl_device_id *devices, cl_context *context,
            cl_command_queue *queue) {
  cl_platform_id platform;
  cl_int err;

  handle(get_platform("NVIDIA", &platform));
  handle(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, NULL));

  *context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
  handle(err);

  *queue = clCreateCommandQueue(*context, *devices, 0, &err);
  handle(err);

  return CL_SUCCESS;
}

png_bytepp sobel_gpu(const png_bytepp img, png_uint_32 height,
                     png_uint_32 width) {
  size_t offset[] = {1, 1};
  size_t global_work_size[] = {height - 1, width - 1};
  size_t size = height * width * 3;

  cl_device_id devices;
  cl_context context;
  cl_command_queue queue;
  cl_int err;

  init(&devices, &context, &queue);

  cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
  handle(err);
  handle(clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, size, img, 0, NULL, NULL));

  cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);
  handle(err);

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernel, NULL, &err);
  handle(err);

  handle(clBuildProgram(program, 1, &devices, NULL, NULL, NULL));
#ifdef DEBUG
  size_t len = 0;
  clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char *buffer = (char *)calloc(len, sizeof(char));
  clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, len, buffer,
                        NULL);
  printf("%s\n", buffer);
  free(buffer);
#endif

  cl_kernel kernel = clCreateKernel(program, "find_edge", &err);
  handle(err);

  handle(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
  handle(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
  handle(clSetKernelArg(kernel, 2, sizeof(unsigned int), &width));
  handle(clSetKernelArg(kernel, 3, sizeof(unsigned int), &height));

  handle(clEnqueueNDRangeKernel(queue, kernel, 2, offset, global_work_size, NULL,
                               0, NULL, NULL));

  auto out = new png_bytep[height];
  for (png_uint_32 h = 0; h < height; h++) out[h] = new png_byte[width];

  handle(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, size, out, 0, NULL, NULL));

  clFlush(queue);
  clFinish(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(d_out);
  clReleaseMemObject(d_in);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseDevice(devices);

  return out;
}