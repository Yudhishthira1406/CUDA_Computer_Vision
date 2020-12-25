#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "ConvNet/ConvNet.h"
#include "Layer_functions/convlayer.h"

#include <cuda.h>
#include <cstdio>

#include <time.h>



static mnist_data *train_set, *test_set;
static unsigned int train_size, test_size;

ConvNet input_layer = ConvNet(0, 0, 28*28);
ConvNet convolve_1 = ConvNet(5*5, 6, 24*24*6);
ConvNet strided_convolve_2 = ConvNet(4*4, 1, 6*6*6);
ConvNet final_3 = ConvNet(6*6*6, 10, 10);

static inline void loaddata()
{
	int ret;
	if(ret = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_size)) {
			printf("error_occured%d", ret);
		}
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_size);
}



void forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	input_layer.reinit();
	convolve_1.reinit();
	strided_convolve_2.reinit();
	final_3.reinit();



	cudaMemcpy(input_layer.output, input, sizeof(float) * 28 * 28, cudaMemcpyHostToDevice);
	
	apply_convolve_1<<<87, 1024>>>((float (*)[28])input_layer.output, (float (*)[24][24])convolve_1.middle, (float (*)[5][5])convolve_1.weight, convolve_1.bias);
	apply_sigmoid<<<6, 1024>>>(convolve_1.middle, convolve_1.output, convolve_1.output_size);

	apply_strided_convolve_2<<<4, 1024>>>((float (*)[24][24])convolve_1.output, (float (*)[6][6])strided_convolve_2.middle, (float (*)[4][4])strided_convolve_2.weight, strided_convolve_2.bias);
	apply_sigmoid<<<1, 1024>>>(strided_convolve_2.middle, strided_convolve_2.output, strided_convolve_2.output_size);

	final_convolve_3<<<3, 1024>>>((float (*)[6][6])strided_convolve_2.output, final_3.middle, (float (*)[6][6][6])final_3.weight, final_3.bias);
    apply_sigmoid<<<1, 10>>>(final_3.middle, final_3.output, final_3.output_size);
    
	cudaDeviceSynchronize();
	
}

void backward_propogation() {
	backpass_final_3<<<3, 1024>>>((float (*)[6][6][6])final_3.d_weight, final_3.d_middle, (float (*)[6][6])strided_convolve_2.output);
	backpass_final_bias_3<<<1, 10>>>(final_3.bias, final_3.d_middle);

	backpass_strided_convolve_2<<<3, 1024>>>((float (*)[6][6])strided_convolve_2.d_output, (float (*)[6][6][6])final_3.weight, final_3.d_middle);
	backpass_strided_convolve_middle_2<<<1, 1024>>>((float (*)[6][6])strided_convolve_2.d_middle, (float (*)[6][6])strided_convolve_2.d_output, (float (*)[6][6])strided_convolve_2.middle);
	backpass_strided_convolve_weight_2<<<4, 1024>>>((float (*)[4][4])strided_convolve_2.d_weight, (float (*)[6][6])strided_convolve_2.d_middle, (float (*)[24][24])convolve_1.output);
	backpass_strided_convolve_bias_2<<<1, 216>>>(strided_convolve_2.bias, (float (*)[6][6])strided_convolve_2.d_middle);

	backpass_convolve_1<<<4, 1024>>>((float (*)[24][24])convolve_1.d_output, (float (*)[4][4])strided_convolve_2.weight, (float (*)[6][6])strided_convolve_2.d_middle);
	backpass_convolve_middle_1<<<6, 1024>>>((float (*)[24][24])convolve_1.d_middle, (float (*)[24][24])convolve_1.d_output, (float (*)[24][24])convolve_1.middle);
	backpas_convolve_weight_1<<<87, 1024>>>((float (*)[5][5])convolve_1.d_weight, (float (*)[24][24])convolve_1.d_middle, (float (*)[28])input_layer.output);
	backpass_convolve_bias_1<<<6, 1024>>>(convolve_1.bias, (float (*)[24][24])convolve_1.d_middle);


	calc_gradient<<<3, 1024>>>(final_3.weight, final_3.d_weight, final_3.filter_size * final_3.num_of_filters);
	calc_gradient<<<1, 16>>>(strided_convolve_2.weight, strided_convolve_2.d_weight, strided_convolve_2.filter_size * strided_convolve_2.num_of_filters);
	calc_gradient<<<1, 150>>>(convolve_1.weight, convolve_1.d_weight, convolve_1.filter_size * convolve_1.num_of_filters);
}

void train()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int epochs = 50;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (epochs-- > 0) {
		err = 0.0f;
		clock_t start, end;
		start = clock();
		for (int i = 0; i < train_size; ++i) {
			float tmp_err;

			forward_pass(train_set[i].data);

			final_3.reinit_backprop();
			strided_convolve_2.reinit_backprop();
			convolve_1.reinit_backprop();

			// Euclid distance of train_set[i]
			calcError<<<1, 10>>>(final_3.d_middle, final_3.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, final_3.d_middle, 1, &tmp_err);
			err += tmp_err;

			backward_propogation();
		}
		end = clock();
		time_taken += double(end - start) / CLOCKS_PER_SEC;
		err /= train_size;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}

int classify(double data[28][28])
{
	float res[10];
	forward_pass(data);
	int max = 0;

	cudaMemcpy(res, final_3.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}
	return max;
}



void predict()
{
	int error = 0;
	for (int i = 0; i < test_size; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_size) * 100.0);
}


int main(int argc, const  char **argv)
{
	srand(time(NULL));

	fprintf(stdout, "%d%d%f", test_size, train_size, threshold);
	loaddata();
	train();
	predict();

	return 0;
}