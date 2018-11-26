#include <iostream>
#include "png.hpp"

int main() {
  auto img = new Image("logo.png");
  std::cout << img->height() << std::endl;
}