#include <exception>
#include <iostream>
#include "png.hpp"
#include "sobel.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./filter [src_img] [dst_img]" << std::endl;
    exit(1);
  }

  Image img(argv[1]);
  try {
    img.transform(sobel_gpu);
  } catch (std::exception& e) {
    std::cout << "Fallback to cpu" << std::endl;
    img.transform(sobel_cpu);
  }
  img.save(argv[2]);
}
