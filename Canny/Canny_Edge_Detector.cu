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

__global__
void non_max_supression(float * magnitude, float * gradient, uchar * output_image, int row, int col){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
       
    if(index < row*col ){
        gradient[index] *= (180*M_1_PI);
        if(gradient[index]<0){gradient[index] += 180;}

        int x = index / col;
        int y = index % col;

        if(x>=1 && y>=1 && x<row-1 && y<col-1 ){

        float q,r;

        if((0<=gradient[index]<22.5) || (157.5<=gradient[index] && gradient[index]<=180))
        {
            q = magnitude[x*col + (y+1)];
            r = magnitude[x*col + (y-1)];
        }else if(22.5<=gradient[index] && gradient[index]<67.5){
            q = magnitude[(x+1)*col + (y-1)];
            r = magnitude[(x-1)*col + (y+1)];
        }else if(67.5<=gradient[index] && gradient[index]<112.5){
            q = magnitude[(x+1)*col + y];
            r = magnitude[(x-1)*col + y];
        }else if(112.5<=gradient[index] && gradient[index]<157.5){
            q = magnitude[(x-1)*col + (y-1)];
            r = magnitude[(x+1)*col + (y+1)];
        }

        if(magnitude[index]>=q && magnitude[index] >=r){
            output_image[index] = magnitude[index];
        }else{
            output_image[index] = 0;
        }
    }

    }
    
}

__global__ 
void double_threshold( uchar * input_image, uchar * output_image,uchar max, int row, int col){
    
    float highThreshold = 0.14 * max;
    float lowThreshold = highThreshold * 0.09;

    uchar weak = 25;
    uchar strong = 255;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < row*col){
    
    if(input_image[index] >= highThreshold) {output_image[index] = strong;}
    else if(input_image[index]<=highThreshold && input_image[index]>=lowThreshold) {output_image[index] = weak;}
    else if(input_image[index]<lowThreshold) {output_image[index] = 0;}
    
    }
}

__global__ 
void hysterisis(uchar * input_image, uchar * output_image, int row, int col){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    uchar weak = 25;
    uchar strong = 255;

    int x = index / col;
    int y = index % col;

    if(x>=1 && y>=1 && x<row-1 && y<col-1 ){

    if(index < row*col){

        if(input_image[index]==weak){
            if(input_image[x*col + (y+1)]==strong || input_image[x*col + (y-1)]==strong || input_image[(x+1)*col + (y-1)]==strong || input_image[(x-1)*col + (y+1)]==strong || input_image[(x+1)*col + y]==strong || input_image[(x-1)*col + y]==strong || input_image[(x-1)*col + (y-1)]==strong || input_image[(x+1)*col + (y+1)]==strong){
                output_image[index] = strong;
            }else{
                output_image[index] = 0;
            }

        }
        else output_image[index] = input_image[index];
    }
   
   }

}

void detect_edge(uchar * input_image, uchar * output_image, int row, int col, int kernel_size, float sigma) {

    //1. Apply Gaussian Blur
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
    
    cudaEvent_t start1,stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    uchar * d_in;
    uchar * d_blur;
    double * d_kernel;

    cudaMalloc((void **) &d_in, sizeof(uchar) * row * col);
    cudaMalloc((void **) &d_blur, sizeof(uchar) * row * col);
    cudaMalloc((void **) &d_kernel, sizeof(double) * kernel_size * kernel_size);

    cudaMemcpy(d_in, input_image, sizeof(uchar) * row * col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(double) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start1);
    apply_gaussian_blur<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_in, d_blur, row, col, d_kernel, kernel_size);
    cudaEventRecord(stop1);

    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start1, stop1);
    cout<<ms1<<endl;
    cudaFree(d_in);
    cudaFree(d_kernel);
  
    //2.Apply Sobel Filter 

    cudaEvent_t start2,stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    float *d_magnitude, *d_gradient; 
    cudaMalloc((void **) &d_magnitude, sizeof(float) * row * col);
    cudaMalloc((void **) &d_gradient, sizeof(float) * row * col);

    cudaEventRecord(start2);
    apply_sobel_filter<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_blur, d_magnitude, d_gradient, row, col);
    cudaEventRecord(stop2);
    
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    cout << ms2 << endl;
    cudaFree(d_blur);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop2);
    
    
    // //3.Non Maximum Suppression

    cudaEvent_t start3,stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    float temp_magnitude[row * col];
    cudaMemcpy(temp_magnitude, d_magnitude, sizeof(float) * row * col, cudaMemcpyDeviceToHost);
    float max = 0.0;
    clock_t begin, end;
    begin = clock();
    for(int i = 0; i < row * col; ++i) {
        if(temp_magnitude[i]>max) {max = temp_magnitude[i];}
    }
    for(int i = 0; i < row * col; ++i) {temp_magnitude[i] = 255.0 * (temp_magnitude[i] / max);}
    end = clock();
    cout << double(end - begin) / double(CLOCKS_PER_SEC) << endl;
    cudaMemcpy(d_magnitude,temp_magnitude,sizeof(float) * row * col,cudaMemcpyHostToDevice);
    
    uchar* output;
    cudaMalloc((void **) &output,sizeof(uchar) * row * col);
    cudaEventRecord(start3);
    non_max_supression<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(d_magnitude,d_gradient,output,row,col);
    cudaEventRecord(stop3);

    float ms3 = 0;
    cudaEventElapsedTime(&ms3,start3,stop3);
    cout<< ms3 <<endl;

    cudaFree(d_magnitude);
    cudaFree(d_gradient);
    
    // //4.Double Threshold

    cudaEvent_t start4,stop4;
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);

    uchar temp_output[row * col];
    cudaMemcpy(temp_output, output, sizeof(uchar) * row * col, cudaMemcpyDeviceToHost);
    uchar max2 = 0;
    clock_t begin2, end2;
    begin2 = clock();
    for(int i = 0; i < row * col; ++i) {
        if(temp_output[i]>max2) {max2 = temp_output[i];}
    }
    end2 = clock();
    cout << double(end2 - begin2) / double(CLOCKS_PER_SEC) << endl;

    uchar * output_threshold;
    cudaMalloc((void **) &output_threshold,sizeof(uchar) * row * col);
    cudaEventRecord(start4);
    double_threshold<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(output,output_threshold,max2,row,col);
    cudaEventRecord(stop4);

    float ms4 = 0;
    cudaEventElapsedTime(&ms4,start4,stop4);
    cout<< ms4 <<endl;

    cudaFree(output);
    
    
    // // 5. Hysterisis

    cudaEvent_t start5,stop5;
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);

    uchar * final_output_image;
    cudaMalloc((void**) &final_output_image,sizeof(uchar) * row * col);
    cudaEventRecord(start5);
    hysterisis<<<(row * col + NUM_OF_THREADS - 1) / NUM_OF_THREADS, NUM_OF_THREADS>>>(output_threshold,final_output_image,row,col);
    cudaEventRecord(stop5);

    float ms5 = 0;
    cudaEventElapsedTime(&ms5,start5,stop5);
    cout<< ms5 <<endl;

    cudaFree(output_threshold);

    
    cudaMemcpy(output_image,final_output_image,sizeof(uchar) * row * col,cudaMemcpyDeviceToHost);
    // int weak = 100, strong = 255;
    // for(int x = 1; x < row - 1; ++x) {
    //     for(int y = 1; y < col - 1; ++y) {
    //         int index = x * col + y;
    //         if(output_image[index]==weak){
    //             if(output_image[x*col + (y+1)]==strong || output_image[x*col + (y-1)]==strong || output_image[(x+1)*col + (y-1)]==strong || output_image[(x-1)*col + (y+1)]==strong || output_image[(x+1)*col + y]==strong || output_image[(x-1)*col + y]==strong || output_image[(x-1)*col + (y-1)]==strong || output_image[(x+1)*col + (y+1)]==strong){
    //                 output_image[index] = strong;
    //             }
    
    //         }
    //     }
    // }
    //cudaFree(final_output_image)

}


