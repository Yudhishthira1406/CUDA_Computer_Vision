#include "ConvNet.h"

ConvNet::ConvNet(int m, int n, int o) {
    filter_size = m;
    num_of_filters = n;
    output_size = o;


    float init_bias[n];
    float init_weight[n * m];

    for(int i = 0; i < n; ++i) {
        init_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        for(int j = 0; j < m; ++j) {
            init_weight[i * m + j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    }

    cudaMalloc(&output, sizeof(float) * output_size);
    cudaMalloc(&middle, sizeof(float) * output_size);
    cudaMalloc(&bias, sizeof(float) * num_of_filters);
    cudaMalloc(&weight, sizeof(float) * filter_size * num_of_filters);
    cudaMalloc(&d_output, sizeof(float) * output_size);
    cudaMalloc(&d_middle, sizeof(float) * output_size);
    cudaMalloc(&d_weight, sizeof(float) * filter_size * num_of_filters);

    cudaMemcpy(bias, init_bias, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, init_weight, sizeof(float) * m * n, cudaMemcpyHostToDevice);
}


ConvNet::~ConvNet() {
    cudaFree(output);
    cudaFree(middle);

    cudaFree(bias);

    cudaFree(weight);

    cudaFree(d_output);
    cudaFree(d_middle);
    cudaFree(d_weight);
}

void ConvNet::reinit() {
    cudaMemset(output, 0, sizeof(float) * output_size);
    cudaMemset(middle, 0, sizeof(float) * output_size);
}

void ConvNet::reinit_backprop() {
    cudaMemset(d_output, 0, sizeof(float) * output_size);
    cudaMemset(d_middle, 0, sizeof(float) * output_size);
    cudaMemset(d_weight, 0, sizeof(float) * filter_size * num_of_filters);
}