#include <iostream>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <math.h>
#include "../image.h"
#include <cuda_runtime.h>
#define NUM_OF_THREADS 1024

#ifndef CANNY_EDGE
#define CANNY_EDGE

void detect_edge(uchar * input_image, uchar * output_image, int row, int col, int kernel_size, float sigma);
__global__ void apply_gaussian_blur(uchar * input_image, uchar * output_image, int row, int col, double * d_kernel, int kernel_size);
__global__ void apply_sobel_filter(uchar * input_image, float * magnitude, float * gradient, int row, int col);
__global__ void non_max_supression(float * magnitude, float * gradient, uchar * output_image, int row, int col);
__global__ void double_threshold(uchar * input_image, uchar * output_image, int row, int col);
__global__ void hysterisis(uchar * input_image, uchar * output_image, int row, int col);

#endif