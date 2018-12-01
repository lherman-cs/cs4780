#pragma once
#include "png.hpp"

png_bytepp sobel_cpu(const png_bytepp img, png_uint_32 height,
                     png_uint_32 width);
png_bytepp sobel_gpu(const png_bytepp img, png_uint_32 height,
                     png_uint_32 width);