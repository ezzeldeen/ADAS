#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>

using namespace cv ;

Mat filter_color (Mat img)
{
    Mat yellow_mask,white_mask,combined;
    inRange(img,Scalar(15,100,120),Scalar(255,255,255),yellow_mask);
    inRange(img,Scalar(0,0,150),Scalar(255,30,255),white_mask);
    combined = white_mask | yellow_mask ;
    return combined;

}

Mat dir_threshold(Mat sobelx, Mat sobely)
{
    Mat magnitude,angles,absx,absy,outDir;
    absx= abs(sobelx);
    absy= abs(sobely);
    cartToPolar( absx, absy, magnitude, angles);
    inRange(angles,0.7,1.2,outDir);
    return outDir;
}
int main()
{
    Mat img,out ;
    img = imread("undistorted.jpg");
    out = filter_color(img);
    namedWindow( "test", CV_WINDOW_AUTOSIZE );
    namedWindow( "original", CV_WINDOW_AUTOSIZE );
    imshow ("test",out);
    imshow("original",img);
    waitKey(0);
    return 0;
}
