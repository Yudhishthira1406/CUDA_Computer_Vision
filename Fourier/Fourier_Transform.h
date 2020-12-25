#include <iostream>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <math.h>
#include "../image.h"
#include <cuda_runtime.h>
#define NUM_OF_THREADS 1024

#ifndef FOURIER_TRANSFORM
#define FOURIER_TRANSFORM

void detect_edge(uchar * input_image, uchar * output_image, int row, int col, int r_in, int r_out);

__global__ void discrete_fourier_transform(uchar * input_image, double * real_image, double * imaginary_image, int row, int col);
__global__ void discrete_fourier_transform2(double * d_real, double * d_imag, double * real_image, double * imaginary_image, int row, int col);
__global__ void fourier_shift(double * real_image, double * imaginary_image, int row, int col);
__global__ void band_pass_filter(double * real_image, double * imaginary_image, int row, int col, int r_in, int r_out);

#endif
