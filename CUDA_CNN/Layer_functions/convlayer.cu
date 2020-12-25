#include "convlayer.h"


__global__ void calc_gradient(float *output, float *grad, int N)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if(pos < N){
		output[pos] += dt * grad[pos];
	}
}

__global__ void apply_convolve_1(float input[28][28], float middle[6][24][24], float weight[6][5][5], float * bias) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 5 * 5 * 6 * 24 * 24;

    if(pos < total_operations) {
        int i1 = (pos /= 1) % 5;
		int i2 = (pos /= 5) % 5;
		int i3 = (pos /= 5) % 6;
		int i4 = (pos /= 6) % 24;
        int i5 = (pos /= 24) % 24;
        
        atomicAdd(&middle[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
        if(i1 == 0 && i2 == 0) {
            middle[i3][i4][i5] += bias[i3];
        }
    }
}

__global__ void apply_strided_convolve_2(float input[6][24][24], float middle[6][6][6], float weight[1][4][4], float * bias) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 4 * 4 * 6 * 6 * 6;

    if(pos < total_operations) {
        int i1 = (pos /= 1) % 4;
		int i2 = (pos /= 4) % 4;
		int i3 = (pos /= 4) % 6;
		int i4 = (pos /= 6) % 6;
		int i5 = (pos /= 6) % 6;

        atomicAdd(&middle[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
        if(i1 == 0 && i2 == 0) {
            middle[i3][i4][i5] += bias[0];
        }
    }
}

__global__ void final_convolve_3(float input[6][6][6], float middle[10], float weight[10][6][6][6], float * bias)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 10 * 6 * 6 * 6;

    if(pos < total_operations) {
        int i1 = (pos /= 1) % 10;
		int i2 = (pos /= 10) % 6;
		int i3 = (pos /= 6) % 6;
        int i4 = (pos /= 6) % 6;
        atomicAdd(&middle[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
        if(i2 == 0 && i3 == 0 && i4 == 0) {
            middle[i1] += bias[i1];
        }
    }
}


__global__ void apply_sigmoid(float * middle, float * output, float output_size) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if(pos < output_size) {
        output[pos] = 1 / (1 + exp(-middle[pos]));
    }
}

__global__ void backpass_final_3(float d_weight[10][6][6][6], float middle[10], float output[6][6][6]) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int total_operations = 10 * 6 * 6 * 6;
    if(pos < total_operations) {
        int i1 = (pos /= 1) % 10;
		int i2 = (pos /= 10) % 6;
		int i3 = (pos /= 6) % 6;
        int i4 = (pos /= 6) % 6;

        d_weight[i1][i2][i3][i4] = middle[i1] * output[i2][i3][i4];
    }
}
__global__ void backpass_final_bias_3(float bias[10], float middle[10]) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int total_operations = 10;
    if(pos < total_operations) {
        bias[pos] += dt * middle[pos];
    }
}

__global__ void backpass_strided_convolve_2(float output[6][6][6], float weight[10][6][6][6], float middle[10]) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 10 * 6 * 6 * 6;
    if (pos < total_operations) {
        int i1 = (pos /= 1) % 10;
		int i2 = (pos /= 10) % 6;
		int i3 = (pos /= 6) % 6;
        int i4 = (pos /= 6) % 6;

        atomicAdd(&output[i2][i3][i4], weight[i1][i2][i3][i4] * middle[i1]);
    }
}

__global__ void backpass_strided_convolve_middle_2(float d_middle[6][6][6], float output[6][6][6], float middle[6][6][6])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int total_operations = 6*6*6;

	if(pos < total_operations){
		
		int i1 = (pos /= 1) % 6;
		int i2 = (pos /= 6) % 6;
		int i3 = (pos /= 6) % 6;

		float sigm = 1 / (1 + exp(-middle[i1][i2][i3]));

		d_middle[i1][i2][i3] = output[i1][i2][i3] * sigm * (1 - sigm);
	}
}

__global__ void backpass_strided_convolve_weight_2(float weight[1][4][4], float middle[6][6][6], float output[6][24][24])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 1*4*4*6*6*6;

	if(pos < total_operations){

		int i1 = (pos /= 1) % 1;
		int i2 = (pos /= 1) % 4;
		int i3 = (pos /= 4) % 4;
		int i4 = (pos /= 4) % 6;
		int i5 = (pos /= 6) % 6;
		int i6 = (pos /= 6) % 6;

		atomicAdd(&weight[i1][i2][i3], middle[i4][i5][i6] * output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void backpass_strided_convolve_bias_2(float bias[1], float middle[6][6][6])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_operations = 6*6*6;
	float d = pow(6.0f, 3.0f);

	if(pos < total_operations) {
		int i1 = (pos /= 1) % 6;
		int i2 = (pos /= 6) % 6;
		int i3 = (pos /= 6) % 6;

		atomicAdd(&bias[0], dt * middle[i1][i2][i3] / d);
	}
}

__global__ void backpass_convolve_1(float output[6][24][24], float weight[1][4][4], float middle[6][6][6])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int total_operations = 1*4*4*6*6*6;

	if(pos < total_operations) {
		int i1 = (pos /= 1) % 1;
		int i2 = (pos /= 1) % 4;
		int i3 = (pos /= 4) % 4;
		int i4 = (pos /= 4) % 6;
		int i5 = (pos /= 6) % 6;
		int i6 = (pos /= 6) % 6;

		atomicAdd(&output[i4][i5 * 4 + i2][i6 * 4 + i3], weight[i1][i2][i3] * middle[i4][i5][i6]);
	}
}

__global__ void backpass_convolve_middle_1(float d_middle[6][24][24], float output[6][24][24], float middle[6][24][24])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int total_operations = 6*24*24;

	if(pos < total_operations) {
		int i1 = (pos /= 1	) % 6;
		int i2 = (pos /= 6	) % 24;
		int i3 = (pos /= 24	) % 24;

		float o = 1 / (1 + exp(-middle[i1][i2][i3]));

		d_middle[i1][i2][i3] = output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void backpas_convolve_weight_1(float weight[6][5][5], float middle[6][24][24], float output[28][28])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int total_operations = 6*5*5*24*24;
	float d = pow(24.0f, 2.0f);

	if(pos < total_operations) {
		int i1 = (pos /= 1) % 6;
		int i2 = (pos /= 6) % 5;
		int i3 = (pos /= 5) % 5;
		int i4 = (pos /= 5) % 24;
		int i5 = (pos /= 24) % 24;

		atomicAdd(&weight[i1][i2][i3], middle[i1][i4][i5] * output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void backpass_convolve_bias_1(float bias[6], float middle[6][24][24])
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int total_operations = 6*24*24;
	float d = pow(24.0f, 2.0f);

	if(pos < total_operations) {
		
		int i1 = (pos /= 1) % 6;
		int i2 = (pos /= 6) % 24;
		int i3 = (pos /= 24) % 24;

		atomicAdd(&bias[i1], dt * middle[i1][i2][i3] / d);
	}
}

__global__ void calcError(float *err, float *output, unsigned int Y, int N)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if(pos < N) {
		err[pos] = ((Y == pos ? 1.0f : 0.0f) - output[pos]);
	}
}


