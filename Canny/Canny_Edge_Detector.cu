
#include "Canny_Edge_Detector.h"



__global__ 
void apply_gaussian_blur(uchar * input_image, uchar * output_image, int row, int col, double * d_kernel, int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < row * col){
        int xind = index / col;
        int yind = index % col;
        int z = kernel_size / 2;
        double kernel_sum = 0.0;
        double val = 0.0;
        for(int i = -z; i <= z; ++i) {
            for(int j = -z; j <= z; ++j) {
                int x1ind = xind + i;
                int y1ind = yind + j;
                if(x1ind >= 0 && x1ind < row && y1ind >= 0 && y1ind < col) {
                    val +=(d_kernel[(kernel_size - 1 - (i + z)) * kernel_size + kernel_size - 1 - (j + z)]) * ((double)input_image[x1ind * col + y1ind]);
                    kernel_sum += d_kernel[(kernel_size - 1 - (i + z)) * kernel_size + kernel_size - 1 - (j + z)];
                }
            }
        }
        
        output_image[index] = val / kernel_sum;
        
        
    }
}



__global__
void apply_sobel_filter(uchar * input_image, float * magnitude, float * gradient, int row, int col) {
    int h_filter[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int v_filter[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < row * col) {
        int xind = index / col;
        int yind = index % col;
        int dX = 0, dY = 0;
        int z = 1;
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                int x1ind = xind + i;
                int y1ind = yind + j;
                if(x1ind >= 0 && x1ind < row && y1ind >= 0 && y1ind < col) {
                    dX += h_filter[((i + z)) * 3 + (j + z)] * input_image[x1ind * col + y1ind];
                    dY += v_filter[((i + z)) * 3 + (j + z)] * input_image[x1ind * col + y1ind];
                }
            }
        }
        magnitude[index] = hypot((float)dX, (float)dY);
        gradient[index] = atan2((float)dY, (float)dX);
    }

}

void detect_edge(uchar * input_image, uchar * output_image, int row, int col, int kernel_size, float sigma) {

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
    
    

    uchar * d_in;
    uchar * d_blur;
    float * d_magnitude, *d_gradient;
    double *d_kernel;
    
    
    cudaMalloc((void **) &d_in, sizeof(uchar) * row * col);
    cudaMalloc((void **) &d_blur, sizeof(uchar) * row * col);
    cudaMalloc((void **) &d_kernel, sizeof(double) * kernel_size * kernel_size);
    cudaMalloc((void **) &d_magnitude, sizeof(float) * row * col);
    cudaMalloc((void **) &d_gradient, sizeof(float) * row * col);

    cudaMemcpy((void *) d_in, input_image, sizeof(uchar) * row * col, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_kernel, kernel, sizeof(double) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stop2);
    cudaEventRecord(start);
    apply_gaussian_blur<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_in, d_blur, row, col, d_kernel, kernel_size);
    
    cudaEventRecord(stop);
    
    apply_sobel_filter<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_blur, d_magnitude, d_gradient, row, col);
    cudaEventRecord(stop2);

    cudaDeviceSynchronize();
    
    cudaEventSynchronize(stop2);
    
    float ms = 0, ms2;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventElapsedTime(&ms2, stop, stop2);
    cout << ms << " " << ms2 << endl;

    float output[row * col];
    cudaMemcpy((void *) output, d_magnitude, sizeof(float) * row * col, cudaMemcpyDeviceToHost);
    float max2 = 0.0;
    clock_t start1, end;
    start1 = clock();
    for(int i = 0; i < row * col; ++i) {
        max2 = max(max2, output[i]);
    }
    for(int i = 0; i < row * col; ++i) {
        output_image[i] = 255.0 * (output[i] / max2);
    }
    end = clock();
    cout << double(end - start1) / double(CLOCKS_PER_SEC) << endl;
    cudaFree(d_in);
    cudaFree(d_blur);
    cudaFree(d_kernel);
    cudaFree(d_magnitude);
    cudaFree(d_gradient);
}



