#include <math.h>
#include "sobel.hpp"

png_bytepp sobel_cpu(const png_bytepp img, png_uint_32 height,
                     png_uint_32 width) {
  png_bytepp dst_img = new png_bytep[height];
  for (auto h = 0; h < height; h++) dst_img[h] = new png_byte[width * 3];

  for (auto r = 1; r < height - 1; r++) {
    for (auto c = 1; c < width - 1; c++) {
      auto gx = img[r - 1][c - 1] - img[r - 1][c + 1] + 2 * img[r][c - 1] -
                2 * img[r][c + 1] + img[r + 1][c - 1] - img[r + 1][c + 1];

      auto gy = img[r - 1][c - 1] + 2 * img[r - 1][c] + img[r - 1][c + 1] -
                img[r + 1][c - 1] - 2 * img[r + 1][c] - img[r + 1][c + 1];

      auto value = 255 - (int)ceil(sqrt(gx * gx + gy * gy));
      dst_img[r][c] = value;
    }
  }
  return dst_img;
}