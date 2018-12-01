# Filter
filter is a simple program that applies sobel edge detection filter to **png files**.

Original:
![original](img/map.png)

Filtered:
![filtered](img/filtered-map.png)

## Dependencies
This repository depends on the following shared libraries:
* OpenCL
* png

## Modules Requires?
filter allows a fallback mechanism to not always use the GPU whenever it's not possible.
To enable the GPU and newest c++17, you need to add a module in Palmetto:

```sh
module load cuda-toolkit gcc/8.2.0
```

## Compilation
Make sure that you have the above dependencies installed. Then, go to the source code
directory and issue:

```sh
make
```

This will compile everything and generates a program called `filter`.

## How to test?

filter requires 2 arguments: src and dst. `src` is the original image path.
`dst` is the output path. 

In the source code directory, there's a sample image file called `map.png` which you can
use to see if this program works by using:

```sh
./filter img/map.png filtered-map.png
```
