#pragma once

// TODO! Add fallback libpng
#include <libpng16/png.h>
#include <string>

typedef png_bytepp (*transform_fn)(const png_bytepp img, png_uint_32 height,
                                   png_uint_32 width);

class Image {
 protected:
  png_infop info;
  png_structp png;
  png_bytepp img;

 public:
  Image(const std::string src);
  ~Image();
  png_uint_32 height() const;
  png_uint_32 width() const;
  png_byte color_type() const;
  png_bytepp data();
  void transform(transform_fn fn);
  void save(const std::string dst);
};
