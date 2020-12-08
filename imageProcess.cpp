#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include "image.h"


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
    int8_pixel pixels[image.rows * image.cols];
    Vec3b v1;
    for(int i = 0; i < image.rows; ++i) {
        for(int j = 0; j < image.cols; ++j) {
            v1 = image.at<Vec3b>(i, j);
            pixels[i * image.cols + j].red = v1.val[2];
            pixels[i * image.cols + j].green = v1.val[1];
            pixels[i * image.cols + j].blue = v1.val[0];
        }
    }
    
    
    return 0;
}
