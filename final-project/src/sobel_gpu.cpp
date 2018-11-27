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
    "__constant sampler_t sampler =\n"
    "      CLK_NORMALIZED_COORDS_FALSE\n"
    "    | CLK_ADDRESS_CLAMP_TO_EDGE\n"
    "    | CLK_FILTER_NEAREST;\n"
    "void __kernel find_edge(__read_only image2d_t in, __write_only image2d_t "
    "out) {\n"
    "const int2 pos = {get_global_id(1), get_global_id(0)};\n"
    "// Compute gradient in +ve x direction\n"
    "const float4 topleft = read_imagef(in, sampler, pos + (int2)(-1, -1));\n"
    "const float4 topright = read_imagef(in, sampler, pos + (int2)(-1, 1));\n"
    "const float4 botleft = read_imagef(in, sampler, pos + (int2)(1, -1));\n"
    "const float4 botright = read_imagef(in, sampler, pos + (int2)(1, 1));\n"
    "float4 gradient_X = topleft\n"
    "- topright\n"
    "+ 2 * read_imagef(in, sampler, pos + (int2)(0, -1))\n"
    "- 2 * read_imagef(in, sampler, pos + (int2)(0, 1))\n"
    "+ botleft\n"
    "- botright;\n"
    "// Compute gradient in +ve y direction\n"
    "float4 gradient_Y = topleft\n"
    "+ 2 * read_imagef(in, sampler, pos + (int2)(-1, 0))\n"
    "+ topright\n"
    "- botleft\n"
    "- 2 * read_imagef(in, sampler, pos + (int2)(1, 0))\n"
    "- botright;\n"
    "float4 value = (float4)1.0 - sqrt(pow(gradient_X, 2) + pow(gradient_Y, "
    "2));\n"
    "write_imagef(out, pos, value);\n"
    "}\n";

void handle(cl_int err) {
  if (err == CL_SUCCESS) return;
  std::__throw_runtime_error(std::to_string(err).data());
}

cl_int get_default_device(cl_platform_id *chosen_platform,
                          cl_device_id *chosen_device) {
  cl_uint num_platforms;
  cl_int ret;

  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if (ret != CL_SUCCESS) return ret;

  cl_platform_id *platforms =
      (cl_platform_id *)calloc(num_platforms, sizeof(cl_platform_id));

  ret = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (ret != CL_SUCCESS) {
    free(platforms);
    return ret;
  }

  size_t size_ret;
  char *_vendor = NULL;
  for (auto i = 0; i < num_platforms; i++) {
    ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, chosen_device,
                         NULL);
    if (ret != CL_SUCCESS) continue;

    ret =
        clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, 0, NULL, &size_ret);
    if (ret != CL_SUCCESS) continue;

    _vendor = (char *)realloc((void *)_vendor, size_ret);
    ret = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, size_ret, _vendor,
                            NULL);
    if (ret != CL_SUCCESS) continue;

    printf("Found %s!\n", _vendor);
    *chosen_platform = platforms[i];
    ret = CL_SUCCESS;
  }

  free(_vendor);
  free(platforms);
  return ret;
}

cl_int init(cl_device_id *devices, cl_context *context,
            cl_command_queue *queue) {
  cl_platform_id platform;
  cl_int err;
  err = get_default_device(&platform, devices);
  handle(err);

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
  cl_image_desc image_desc;
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc.image_width = width;
  image_desc.image_height = height;
  image_desc.image_array_size = 1;
  image_desc.image_row_pitch = 0;
  image_desc.image_slice_pitch = 0;
  image_desc.num_mip_levels = 0;
  image_desc.num_samples = 0;
  image_desc.buffer = NULL;

  cl_device_id devices;
  cl_context context;
  cl_command_queue queue;
  cl_int err;

  init(&devices, &context, &queue);

  const cl_image_format format = {CL_INTENSITY, CL_UNORM_INT8};
  cl_mem d_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              &format, &image_desc, (void *)img, &err);
  handle(err);

  cl_mem d_out = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &image_desc,
                               NULL, &err);
  handle(err);

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernel, NULL, &err);
  handle(err);

  err = clBuildProgram(program, 1, &devices, NULL, NULL, NULL);
  handle(err);
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

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
  handle(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
  handle(err);

  err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, global_work_size, NULL,
                               0, NULL, NULL);
  handle(err);

  auto out = new png_bytep[height];
  for (auto h = 0; h < height; h++) out[h] = new png_byte[width];

  const size_t origin[] = {0, 0, 0};
  const size_t region[] = {width, height, 1};
  err = clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, out, 0,
                           NULL, NULL);
  handle(err);

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