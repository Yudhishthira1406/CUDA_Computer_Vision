#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include "image.h"
#include "Canny/Canny_Edge_Detector.h"


using namespace std;
using namespace cv;


int main(int argc, char const *argv[])
{
    if(argc <= 1) {
        cout << "No file input";
        return 0;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);
    if(!image.data) {
        cout << "Incorrect path specified";
        return 0;
    }
    int row = image.rows;
    int col = image.cols;
    int8_pixel pixels[row * col];
    Vec3b v1;
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            v1 = image.at<Vec3b>(i, j);
            pixels[i * col + j].red = v1.val[2];
            pixels[i * col + j].green = v1.val[1];
            pixels[i * col + j].blue = v1.val[0];
        }
    }

    int8_pixel output_image[row * col];
    apply_gaussian_blur(pixels, output_image, row, col, 5, 1);
    cout << row << " " << col << " " << image.channels() << endl;
    Mat final_image(Size(col, row), CV_8SC3);
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            Vec3b &v = final_image.at<Vec3b>(i, j);
            v.val[0] = output_image[i * col + j].blue;
            v.val[1] = output_image[i * col + j].green;
            v.val[2] = output_image[i * col + j].red;
        }
    }
    cout << final_image.rows << " " << final_image.cols << " " << final_image.channels() << endl;
    Vec3b v = final_image.at<Vec3b>(row - 300, col - 1);
    cout << (int)v.val[0] << " " << (int)v.val[1] << " " << (int)v.val[2] << endl;
    cout << output_image[(row -299) * col - 1].blue << " " << output_image[(row -299) * col - 1].green << " " << output_image[(row -299) * col - 1].red << endl;

    namedWindow("FINAL");
    imshow("FINAL", final_image);
    imwrite("test.png", final_image);
    waitKey(0);

    
    
    return 0;
}
