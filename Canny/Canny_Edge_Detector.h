#include <iostream>
#include <math.h>
#include <iomanip>
#include <math.h>
#include "../image.h"
#include <cuda_runtime.h>
#define NUM_OF_THREADS 1024

#ifndef CANNY_EDGE
#define CANNY_EDGE

void detect_edge(int8_pixel * input_image, int row, int col);
void apply_gaussian_blur(int8_pixel * input_image, int8_pixel * output_image, int row, int col, int kernel_size, int sigma);
void apply_sobel_filter(int8_pixel * input_image, float * magnitude, float * gradient, int row, int col);
__global__ void calculate_magnitude_and_gradient(int8_pixel * in_x, int8_pixel * in_y, float * magnitude, float * gradient, int row, int col);
__global__ void convolve(int8_pixel * input_image, int8_pixel * output_image, int row, int col, double* d_kernel, int kernel_size, char c);
__global__ void non_max_supression(float * magnitude, float * gradient, int * output_image, int row, int col);
__global__ void low_threshold(int8_pixel * input_image, int8_pixel * output_image, int row, int col);
__global__ void high_threshold(int8_pixel * imput_image, int8_pixel * output_image, int row, int col);
__global__ void hysterisis(int8_pixel * input_image, int8_pixel * output_image, int row, int col);


#endif