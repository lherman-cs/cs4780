#include "png.hpp"
#include <stdio.h>

Image::Image(const std::string src) {
  FILE *fp = fopen(src.data(), "rb");

  this->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!this->png) std::__throw_invalid_argument("Can't create image struct");

  this->info = png_create_info_struct(this->png);
  if (!this->info) std::__throw_runtime_error("Can't create image info");

  png_init_io(this->png, fp);
  png_read_info(this->png, this->info);

  auto color_type = this->color_type();
  if (color_type == PNG_COLOR_TYPE_RGB ||
      color_type == PNG_COLOR_TYPE_RGB_ALPHA)
    png_set_rgb_to_gray_fixed(this->png, 1, -1, -1);

  png_read_update_info(this->png, this->info);

  auto height = this->height();
  auto width = this->width();
  this->img = new png_bytep[height];
  for (auto h = 0; h < height; h++) this->img[h] = new png_byte[width];

  png_read_image(this->png, this->img);
  fclose(fp);
}

Image::~Image() {
  auto height = this->height();
  for (auto h = 0; h < height; h++) delete[] this->img[h];
  delete[] this->img;

  png_destroy_read_struct(&this->png, &this->info, NULL);
}

png_uint_32 Image::height() const {
  return png_get_image_height(this->png, this->info);
}

png_uint_32 Image::width() const {
  return png_get_image_width(this->png, this->info);
}

png_byte Image::color_type() const {
  return png_get_color_type(this->png, this->info);
}

png_bytepp Image::data() { return this->img; }

void Image::transform(transform_fn fn) {
  auto height = this->height();
  auto width = this->width();
  auto transformed = fn(this->img, height, width);
  for (auto h = 0; h < height; h++) delete[] this->img[h];
  delete[] this->img;

  this->img = transformed;
}

void Image::save(const std::string dst) {
  FILE *fp = fopen(dst.data(), "wb");

  auto dst_png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!dst_png) std::__throw_invalid_argument("Can't create image struct");

  auto dst_info = png_create_info_struct(dst_png);
  if (!dst_info) std::__throw_runtime_error("Can't create image info");

  png_init_io(dst_png, fp);

  png_uint_32 width, height;
  int color_depth, color_type, interlace_type, compression_type, filter_method;
  png_get_IHDR(this->png, this->info, &width, &height, &color_depth,
               &color_type, &interlace_type, &compression_type, &filter_method);

  png_set_IHDR(dst_png, dst_info, width, height, color_depth, color_type,
               interlace_type, compression_type, filter_method);

  png_write_info(dst_png, dst_info);
  png_write_image(dst_png, this->img);

  png_destroy_write_struct(&dst_png, &dst_info);

  fclose(fp);
}
