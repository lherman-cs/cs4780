#pragma once

// TODO! Add fallback libpng
#include <libpng16/png.h>
#include <string>

class Image {
 private:
  png_infop info;
  png_structp png;
  png_bytepp img;

 public:
  Image(const std::string src);
  png_uint_32 height();
  png_uint_32 width();
  png_byte color_type();
};