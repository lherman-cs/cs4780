#include <iostream>
#include "png.hpp"
#include "sobel.hpp"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./filter [src_img] [dst_img]" << std::endl;
    exit(1);
  }

  Image img(argv[1]);
  img.transform(sobel);
  img.save(argv[2]);
}
