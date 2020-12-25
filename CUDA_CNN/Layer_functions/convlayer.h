#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

__global__ void calc_gradient(float *output, float *grad, const int N);
__global__ void apply_sigmoid(float * middle, float * output, float output_size); 
__global__ void calcError(float *err, float *output, unsigned int Y, const int N);

__global__ void apply_convolve_1(float input[28][28], float middle[6][24][24], float weight[6][5][5], float * bias);
__global__ void backpass_convolve_1(float output[6][24][24], float weight[1][4][4], float middle[6][6][6]);
__global__ void backpass_convolve_middle_1(float d_middle[6][24][24], float output[6][24][24], float middle[6][24][24]);
__global__ void backpas_convolve_weight_1(float weight[6][5][5], float middle[6][24][24], float output[28][28]);
__global__ void backpass_convolve_bias_1(float bias[6], float middle[6][24][24]);

__global__ void apply_strided_convolve_2(float input[6][24][24], float middle[6][6][6], float weight[1][4][4], float * bias);;
__global__ void backpass_strided_convolve_2(float output[6][6][6], float weight[10][6][6][6], float middle[10]);
__global__ void backpass_strided_convolve_middle_2(float d_middle[6][6][6], float output[6][6][6], float middle[6][6][6]);
__global__ void backpass_strided_convolve_weight_2(float weight[1][4][4], float middle[6][6][6], float output[6][24][24]);
__global__ void backpass_strided_convolve_bias_2(float bias[1], float middle[6][6][6]);

__global__ void final_convolve_3(float input[6][6][6], float middle[10], float weight[10][6][6][6], float * bias);
__global__ void backpass_final_3(float d_weight[10][6][6][6], float middle[10], float output[6][6][6]);
__global__ void backpass_final_bias_3(float bias[10], float middle[10]);