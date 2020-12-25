#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include "image.h"
// #include "Canny/Canny_Edge_Detector.h"
#include "Fourier/Fourier_Transform.h"


using namespace std;
using namespace cv;

uchar output_image[10000000];



int main(int argc, char const *argv[])
{
    if(argc <= 1) {
        cout << "No file input";
        return 0;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if(!image.data) {
        cout << "Incorrect path specified";
        return 0;
    }
    namedWindow("IN");
    imshow("IN", image);
    int row = image.rows;
    int col = image.cols;
    uchar *pixels = image.isContinuous()?image.data:image.clone().data;

    // uchar output_image[row * col];
    detect_edge(pixels, output_image, row, col, -1, 40);
    cout << row << " " << col << " " << image.channels() << endl;
    Mat final_image(row, col, CV_8UC1, &output_image);
    // for(int i = 0; i < row; ++i) {
    //     for(int j = 0; j < col; ++j) {
    //         Vec3b &v = final_image.at<Vec3b>(i, j);
    //         v.val[0] = output_image[i * col + j].blue;
    //         v.val[1] = output_image[i * col + j].green;
    //         v.val[2] = output_image[i * col + j].red;
    //     }
    // }
    // cout << final_image.rows << " " << final_image.cols << " " << final_image.channels() << endl;
    // Vec3b v = final_image.at<Vec3b>(row - 300, col - 1);
    // cout << (int)v.val[0] << " " << (int)v.val[1] << " " << (int)v.val[2] << endl;
    // cout << output_image[(row -299) * col - 1].blue << " " << output_image[(row -299) * col - 1].green << " " << output_image[(row -299) * col - 1].red << endl;

    namedWindow("FINAL");
    imshow("FINAL", final_image);
    imwrite("test.png", final_image);
    waitKey(0);

    
    
    return 0;
}

