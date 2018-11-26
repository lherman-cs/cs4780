#include "png.hpp"
#include <iostream>

int main(){
    auto img = new Image("logo.png");
    std::cout << img->height() << std::endl; 
}