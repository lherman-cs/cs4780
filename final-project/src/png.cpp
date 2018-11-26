#include "png.hpp"
#include <stdio.h>
#include <iostream>

Image::Image(const std::string src) {
  FILE *fp = fopen(src.data(), "rb");

  auto transform = PNG_TRANSFORM_IDENTITY;
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

  this->img = new png_bytep[this->height()];
  png_read_image(this->png, this->img);

  fclose(fp);
}

png_uint_32 Image::height() {
  return png_get_image_height(this->png, this->info);
}

png_uint_32 Image::width() {
  return png_get_image_width(this->png, this->info);
}

png_byte Image::color_type() {
  return png_get_color_type(this->png, this->info);
}