#include "Fourier_Transform.h"

double Re[1000000], Im[1000000];
__global__ 
void discrete_fourier_transform(uchar * input_image, double * real_image, double * imaginary_image, int row, int col) {
    __shared__ double cache1[1024], cache2[1024];
    int index = blockIdx.x;
    int row_num = threadIdx.x;
    int xind = index / col;
    int yind = index % col;
    double t1 = 0.0, t2 = 0.0;
    while(row_num < row) {

        t1 += cos(2 * M_PI * xind * row_num / row) * input_image[row_num * col + yind];
        t2 += -sin(2 * M_PI * xind * row_num / row) * input_image[row_num * col + yind];
        row_num += blockDim.x;
    }
    cache1[threadIdx.x] = t1 ;
    cache2[threadIdx.x] = t2 ;
    
    __syncthreads();

    int i = blockDim.x / 2;
    int cacheindex = threadIdx.x;
    while(i != 0) {
        if(cacheindex < i) {
            cache1[cacheindex] += cache1[cacheindex + i];
            cache2[cacheindex] += cache2[cacheindex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheindex == 0) {
        real_image[blockIdx.x] = cache1[0];
        imaginary_image[blockIdx.x] = cache2[0];
    }

}

__global__ 
void discrete_fourier_transform2(double * d_real, double * d_imag, double * real_image, double * imaginary_image, int row, int col) {
    __shared__ double cache1[1024], cache2[1024];
    int index = blockIdx.x;
    int row_num = threadIdx.x;
    int xind = index / col;
    int yind = index % col;
    double t1 = 0.0, t2 = 0.0;
    while(row_num < col) {

        t1 += d_real[xind * col + row_num] * cos(2 * M_PI * row_num * yind/col) + d_imag[xind * col + row_num] * sin(2 * M_PI * row_num * yind / col);
        t2 += d_imag[xind * col + row_num] * cos(2 * M_PI * row_num * yind/col) - d_real[xind * col + row_num] * sin(2 * M_PI * row_num * yind / col);
        row_num += blockDim.x;
    }
    cache1[threadIdx.x] = t1 ;
    cache2[threadIdx.x] = t2 ;
    
    __syncthreads();

    int i = blockDim.x / 2;
    int cacheindex = threadIdx.x;
    while(i != 0) {
        if(cacheindex < i) {
            cache1[cacheindex] += cache1[cacheindex + i];
            cache2[cacheindex] += cache2[cacheindex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheindex == 0) {
        real_image[blockIdx.x] = cache1[0] / sqrt(float(row * col));
        imaginary_image[blockIdx.x] = cache2[0] / sqrt(float(row * col));
    }

}

__global__
void fourier_shift(double * real_image, double * imaginary_image, int row, int col) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < row * col / 2) {
        int xind = index / col;
        int yind = index % col;
        xind = xind + row / 2;
        yind = (yind + col / 2) % col;
        double temp = real_image[index];
        real_image[index] = real_image[xind * col + yind];
        real_image[xind * col + yind] = temp;
        temp = imaginary_image[index];
        imaginary_image[index] = imaginary_image[xind * col + yind];
        imaginary_image[xind * col + yind] = temp;

    }
}

__global__ 
void band_pass_filter(double * real_image, double * imaginary_image, int row, int col, int r_in, int r_out) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < row * col) {
        int xind = index / col;
        int yind = index % col;
        double dist = (xind - row / 2) * (xind - row / 2) + (yind - col / 2) * (yind - col / 2);
        if((dist <= r_out * r_out)) {
            real_image[xind * col + yind] = 0;
            imaginary_image[xind * col + yind] = 0;
        }

    }
}

__global__
void inverse_discrete_transform(double * d_real, double * d_imag, double * real_image, double * imaginary_image, int row, int col) {
    __shared__ double cache1[1024], cache2[1024];
    int index = blockIdx.x;
    int row_num = threadIdx.x;
    int xind = index / col;
    int yind = index % col;
    double t1 = 0.0, t2 = 0.0;
    while(row_num < row) {

        t1 += cos(2 * M_PI * xind * row_num / row) * d_real[row_num * col + yind] - sin(2 * M_PI * xind * row_num / row) * d_imag[row_num * col + yind];
        t2 += sin(2 * M_PI * xind * row_num / row) * d_real[row_num * col + yind] + cos(2 * M_PI * xind * row_num / row) * d_imag[row_num * col + yind];
        row_num += blockDim.x;
    }
    cache1[threadIdx.x] = t1;
    cache2[threadIdx.x] = t2;
    
    __syncthreads();

    int i = blockDim.x / 2;
    int cacheindex = threadIdx.x;
    while(i != 0) {
        if(cacheindex < i) {
            cache1[cacheindex] += cache1[cacheindex + i];
            cache2[cacheindex] += cache2[cacheindex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheindex == 0) {
        real_image[blockIdx.x] = cache1[0];
        imaginary_image[blockIdx.x] = cache2[0];
    }
}

__global__ 
void inverse_discrete_transform2(double * d_real, double * d_imag, double * real_image, double * imaginary_image, int row, int col) {
    __shared__ double cache1[1024], cache2[1024];
    int index = blockIdx.x;
    int row_num = threadIdx.x;
    int xind = index / col;
    int yind = index % col;
    double t1 = 0.0, t2 = 0.0;
    while(row_num < col) {

        t1 += d_real[xind * col + row_num] * cos(2 * M_PI * row_num * yind/col) - d_imag[xind * col + row_num] * sin(2 * M_PI * row_num * yind / col);
        t2 += d_imag[xind * col + row_num] * cos(2 * M_PI * row_num * yind/col) + d_real[xind * col + row_num] * sin(2 * M_PI * row_num * yind / col);
        row_num += blockDim.x;
    }
    cache1[threadIdx.x] = t1;
    cache2[threadIdx.x] = t2;
    
    __syncthreads();

    int i = blockDim.x / 2;
    int cacheindex = threadIdx.x;
    while(i != 0) {
        if(cacheindex < i) {
            cache1[cacheindex] += cache1[cacheindex + i];
            cache2[cacheindex] += cache2[cacheindex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheindex == 0) {
        real_image[blockIdx.x] = cache1[0] / sqrt(float(row * col));
        imaginary_image[blockIdx.x] = cache2[0] / sqrt(float(row * col));
    }

}

void detect_edge(uchar * input_image, uchar * output_image, int row, int col, int r_in, int r_out) {

    uchar * d_in;
    double * X_real, * X_imag;
    double * X_Re, *X_Im;
    double * X_RE, *X_IM;

    cudaMalloc((void **) &d_in, sizeof(uchar) * row * col);
    cudaMalloc((void **) &X_real, sizeof(double) * row * col);
    cudaMalloc((void **) &X_imag, sizeof(double) * row * col);
    cudaMalloc((void **) &X_Re, sizeof(double) * row * col);
    cudaMalloc((void **) &X_Im, sizeof(double) * row * col);
    cudaMalloc((void **) &X_RE, sizeof(double) * row * col);
    cudaMalloc((void **) &X_IM, sizeof(double) * row * col);

    cudaMemcpy((void *) d_in, input_image, sizeof(uchar) * row * col, cudaMemcpyHostToDevice);
    clock_t start, end;
    start = clock();
    discrete_fourier_transform<<< row * col, NUM_OF_THREADS >>>(d_in, X_real, X_imag, row, col);
    discrete_fourier_transform2<<< row * col, NUM_OF_THREADS >>>(X_real, X_imag, X_Re, X_Im, row, col);
    fourier_shift<<< (row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS >>>(X_Re, X_Im, row, col);
    band_pass_filter<<< (row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS >>>(X_Re, X_Im, row, col, r_in, r_out);
    fourier_shift<<< (row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS >>>(X_Re, X_Im, row, col);
    inverse_discrete_transform<<< row * col, NUM_OF_THREADS >>>(X_Re, X_Im, X_real, X_imag, row, col);
    inverse_discrete_transform2 <<< row * col, NUM_OF_THREADS >>>(X_real, X_imag, X_RE, X_IM, row, col);

    cudaDeviceSynchronize();
    end = clock();
    cout << double(end - start) / double(CLOCKS_PER_SEC) << endl;

    
    cudaMemcpy(Re, X_RE, sizeof(double) * row * col, cudaMemcpyDeviceToHost);
    cudaMemcpy(Im, X_IM, sizeof(double) * row * col, cudaMemcpyDeviceToHost);

    for(int i = 0; i < row * col; ++i) output_image[i] = sqrt(Re[i] * Re[i] + Im[i] * Im[i]);

    
}

