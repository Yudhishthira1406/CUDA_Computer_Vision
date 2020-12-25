#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>



#ifndef CONV_NET_H
#define CONV_NET_H
#endif

struct ConvNet {
	int filter_size;
    int num_of_filters;
    int output_size;

	float *output;
	float *middle;

	float *bias;
	float *weight;

	float *d_output;
	float *d_middle;
	float *d_weight;

	ConvNet(int m, int n, int o);

	~ConvNet();

	void reinit();
	void reinit_backprop();
};
