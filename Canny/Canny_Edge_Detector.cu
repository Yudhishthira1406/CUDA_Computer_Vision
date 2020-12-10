
#include "Canny_Edge_Detector.h"

__global__ 
void convolve(int8_pixel * input_image, int8_pixel * output_image, int row, int col, double * d_kernel, int kernel_size, char c) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < row * col){
        int xind = index / col;
        int yind = index % col;
        int z = kernel_size / 2;
        double kernel_sum = 0.0;
        double red = 0.0;
        double blue = 0.0;
        double green = 0.0;
        for(int i = -z; i <= z; ++i) {
            for(int j = -z; j <= z; ++j) {
                int x1ind = xind + i;
                int y1ind = yind + j;
                if(x1ind >= 0 && x1ind < row && y1ind >= 0 && y1ind < col) {
                    red +=(d_kernel[(i + z) * kernel_size + (j + z)]) * ((double)input_image[x1ind * col + y1ind].red);
                    blue +=(d_kernel[(i + z) * kernel_size + (j + z)]) *((double)input_image[x1ind * col + y1ind].blue);
                    green += (d_kernel[(i + z) * kernel_size + (j + z)]) *((double)input_image[x1ind * col + y1ind].green);
                    kernel_sum += d_kernel[(i + z) * kernel_size + (j + z)];
                }
            }
        }
        if(c == 'G') {
            output_image[index].red = red / kernel_sum;
            output_image[index].blue = blue / kernel_sum;
            output_image[index].green = green / kernel_sum;
        }
        
    }
}

void apply_gaussian_blur(int8_pixel * input_image, int8_pixel * output_image, int row, int col, int kernel_size, int sigma) {
    cout << setprecision(15);
    double kernel[kernel_size * kernel_size];
    for(int i = 0; i < kernel_size; ++i) {
        for(int j = 0; j < kernel_size; ++j) {
            int xC = pow((i - (kernel_size + 1) / 2), 2);
            int yC = pow((j - (kernel_size + 1) / 2), 2);
            double sqsigma = pow(sigma, 2);
            double val = exp(-(xC + yC) / (2 * sqsigma));
            val = val * ((M_1_PI) / (2 * sqsigma));
            kernel[i * kernel_size + j] = val;
            kernel[i * kernel_size + j] /= kernel[0];

        }
        
    }
    // for(int i = 0; i < kernel_size * kernel_size;++i) cout << kernel[i] << " " << (int)input_image[i].red << endl;
    //GPU variables
    int8_pixel *d_in;
    int8_pixel *d_out;
    double * d_kernel;
    //Time
    cudaEvent_t start, stop;
    float el;


    //Assigning GPU memory
    cudaMalloc((void **) &d_in, sizeof(int8_pixel) * row * col);
    cudaMalloc((void **) &d_out, sizeof(int8_pixel) * row * col);
    cudaMalloc((void **) &d_kernel, sizeof(kernel[0]) * kernel_size * kernel_size);
    //Copying data from CPU to GPU
    cudaMemcpy((void *)d_in, (void *)input_image, sizeof(input_image[0]) * row * col, cudaMemcpyHostToDevice);
    // for(int i = 0; i < 10; ++i) cout << d_in[i].red << endl;
    
    cudaMemcpy((void *)d_kernel, (void *)kernel, sizeof(kernel[0]) * kernel_size * kernel_size, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    convolve<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_in, d_out, row, col, d_kernel, kernel_size, 'G');
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&el, start, stop);
    // convolve<<<1, 1>>>(d_in, d_out, row, col, d_kernel, kernel_size, 'G');
    cudaDeviceSynchronize();
    // for(int i = 0; i < 10; ++i) {
    //     cout << input_image[i].red << " " << output_image[i].red << endl;
    // }

    cudaMemcpy((void *)output_image, (void *)d_out, sizeof(int8_pixel) * row * col, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    cout << el << endl;
}

__global__ 
void calculate_magnitude_and_gradient(int8_pixel * in_x, int8_pixel * in_y, float * magnitude, float * gradient, int row, int col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < row * col) {
        float dx = (0.2989 * (float)in_x.red + 0.5870 * (float)in_x.green + 0.1140 * (float)in_x.blue);
        float dy = (0.2989 * (float)in_y.red + 0.5870 * (float)in_y.green + 0.1140 * (float)in_y.blue);
        magnitude[index] = sqrt(dx*dx + dy*dy);
        gradient[index] = atan(dy/dx);
    }
}

void apply_sobel_filter(int8_pixel * input_image, float * magnitude, float * gradient, int row, int col) {
    
    int8_pixel * horiz, *vert, *d_in;
    double *d_h_kernel, *d_v_kernel;
    float *d_mag, *d_grad;
    double vert_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    double horiz_kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    cudaMalloc((void **) &horiz, sizeof(int8_pixel) * row * col);
    cudaMalloc((void **) &vert, sizeof(int8_pixel) * row * col);
    cudaMalloc((void **) &d_h_kernel, sizeof(double) * 9);
    cudaMalloc((void **) &d_v_kernel, sizeof(double) * 9);
    cudaMalloc((void **) &d_mag, sizeof(float) * row * col);
    cudaMalloc((void **) &d_grad, sizeof(float) * row * col);

    cudaMemcpy((void *) d_in, (void *) input_image, sizeof(int8_pixel) * row * col, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_h_kernel, (void *) horiz_kernel, sizeof(double) * row * col, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_v_kernel, (void *) vert_kernel, sizeof(double) * row * col, cudaMemcpyHostToDevice);

    convolve<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_in, horiz, row, col, d_h_kernel, 3, 'S');
    convolve<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_in, vert, row, col, d_v_kernel, 3, 'S');

    calculate_magnitude_and_gradient<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(horiz, vert, d_mag, d_grad, row, col);
    cudaDeviceSynchronize();

    cudaMemcpy((void *) magnitude, (void *) d_mag, sizeof(float) * row * col, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) gradient, (void *) d_grad, sizeof(float) * row * col, cudaMemcpyDeviceToHost);

    cudaFree(horiz);
    cudaFree(vert);
    cudaFree(d_h_kernel);
    cudaFree(d_v_kernel);
    cudaFree(d_mag);
    cudaFree(d_grad);

}



