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

Mat magnitude_threshold(Mat sobelx , Mat sobely)
{
    Mat mag,scaledMag,outMag;
    double maxValue,minValue;
    magnitude( sobelx, sobely, mag);
    minMaxLoc(mag, &minValue, &maxValue);
    divide(maxValue, mag, scaledMag);
    inRange(scaledMag,0.5,4,outMag);
    minMaxLoc(outMag, &minValue, &maxValue);
    return outMag;


}

void searchForLanes(Mat img)
{
    Mat outImg,halfImg,hist ,midHist;
    double maxVal,minVal;
    Point leftPoint,rightPoint,minInd;
    int nimages = 1;
    int channels[] = {0} ;
    int dims = 1;
    int histSize[] = {256} ;
    float hranges[] = { 0, 256 };
    const float *ranges[] = {hranges};
    cvtColor(img,outImg,CV_GRAY2RGB);
    halfImg = img (Range(0,img.rows/2),Range(0,img.cols));
    calcHist(&img,
    nimages,
    channels,
    Mat(), // No mask
    hist, dims, histSize, ranges,true,false);
    int midPoint = hist.rows/2;
    midHist =hist(Range(0,midPoint),Range(0,1));
    minMaxLoc(midHist,&minVal,&maxVal,&minInd,&leftPoint);
    //std::cout<<maxInd<<"\n";
    midHist = hist(Range(midPoint,hist.rows),Range(0,1));
    minMaxLoc(midHist,&minVal,&maxVal,&minInd,&rightPoint);
    //std::cout<<midHist<<"\n";
    //std::cout<<hist.rows<<" "<<hist.cols<<"\n";
    int numOfWindows = 9;
    int windowsHeight = img.rows / numOfWindows ;
    Mat nonZeroCoordinates,nonZeroX,nonZeroY;
    findNonZero(img, nonZeroCoordinates);
    nonZeroX = nonZeroCoordinates[1];
    nonZeroY = nonZeroCoordinates[0];
    int margin = 100 , minPix = 50 ;



}

int main()
{
    Mat img,outColor,grad_x,grad_y,outDir,outMag,combined ;
    img = imread("undistorted.jpg");
    outColor = filter_color(img);
    Sobel( outColor, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    Sobel( outColor, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    outDir = dir_threshold(grad_x,grad_y);
    outMag = magnitude_threshold(grad_x,grad_y);
    combined = (outColor & (outMag | outDir));
    searchForLanes(combined);
    namedWindow( "test", CV_WINDOW_AUTOSIZE );
    namedWindow( "original", CV_WINDOW_AUTOSIZE );
    imshow ("test",combined);
    imshow("original",img);
    waitKey(0);
    return 0;
}
