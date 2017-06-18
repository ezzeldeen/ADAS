#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <vector>
#include <string>


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

Mat warp_image(Mat img, Point2f* src_vertices , Point2f* dst_vertices)
{
    Mat M,warped;
    M = getPerspectiveTransform(src_vertices,dst_vertices);
    warpPerspective(img,warped, M,warped.size());
    return warped;

}


void searchForLanes(Mat img)
{
    Mat outImg,halfImg,hist ,midHist;
    double maxVal,minVal;
    Point leftPoint,rightPoint,minInd;
    cvtColor(img,outImg,CV_GRAY2RGB);
    halfImg = img (Range(img.rows/2,img.rows),Range(0,img.cols));
    reduce(halfImg,hist,0,CV_REDUCE_SUM, CV_32SC1);
    midHist =hist(Range(0,1),Range(0,hist.cols/2));
    minMaxLoc(midHist,&minVal,&maxVal,&minInd,&leftPoint);
    midHist = hist(Range(0,1),Range(hist.cols/2,hist.cols));
    minMaxLoc(midHist,&minVal,&maxVal,&minInd,&rightPoint);
    int left_point = leftPoint.x , right_point = rightPoint.x+(img.cols/2);
    int numOfWindows = 9,margin=50;
    int window_height = img.rows / numOfWindows ;
    Mat nonZeroCoordinates;
    vector<cv::Point> leftLane;
    vector<cv::Point> rightLane;
    vector<cv::Point> current_nonZero;
    int win_y_low, win_y_high, win_left_x_low, win_left_x_high, win_right_x_low, win_right_x_high;
    for(int i = 0 ; i < 2; i++)
    {
        win_y_low = img.rows - ((i+1)*window_height);
        win_y_high = win_y_low + window_height ;
        win_left_x_low = left_point - margin ;
        win_left_x_high = left_point + margin ;
        win_right_x_low = right_point - margin ;
        win_right_x_high = right_point + margin;
        if(countNonZero(img(Range(win_y_low,win_y_high),Range(win_left_x_low,win_left_x_high))))
        {
            findNonZero(img(Range(win_y_low,win_y_high),Range(win_left_x_low,win_left_x_high)),current_nonZero);
            leftLane.insert(leftLane.end(), current_nonZero.begin(), current_nonZero.end());
            current_nonZero.clear();
        }
        if(countNonZero(img(Range(win_y_low,win_y_high),Range(win_right_x_low,win_right_x_high))))
        {
            findNonZero(img(Range(win_y_low,win_y_high),Range(win_right_x_low,win_right_x_high)),current_nonZero);
            rightLane.insert(rightLane.end(), current_nonZero.begin(), current_nonZero.end());
            current_nonZero.clear();
        }
        leftLane.insert(leftLane.end(), rightLane.begin(), rightLane.end());
        const cv::Point *pts = (const cv::Point*) Mat(leftLane).data;
        int npts = Mat(leftLane).rows;
        std::cout<<"\n";
        std::cout<<npts;
        imshow("yaraaaaaaaaaab",img);
        //fillPoly(img,&pts,&npts,1, Scalar( 255, 255, 255 ), 8);
        //imshow("yaraaaaaaaaaaab",img);

    }

}




int main()
{
    Mat img,res,outColor,grad_x,grad_y,outDir,outMag,combined,warped;
    img = imread("undistorted.jpg");
    resize(img, res, Size(img.cols * 0.5,img.rows * 0.5), 0, 0, CV_INTER_CUBIC);
    outColor = filter_color(res);
    Sobel( outColor, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    Sobel( outColor, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    outDir = dir_threshold(grad_x,grad_y);
    outMag = magnitude_threshold(grad_x,grad_y);
    combined = (outColor & (outMag | outDir));
    Point2f src[4],dst[4];
    src[0]=Point2f( 290, 230 );
    src[1]=Point2f( 350, 230 );
    src[2]=Point2f( 520, 340 );
    src[3]=Point2f( 130, 340 );
    dst[0]= Point2f( 130, 0 );
    dst[1]= Point2f( 520, 0 );
    dst[2]= Point2f( 520, 310);
    dst[3]= Point2f( 130, 310 );
    warped = warp_image(combined,src,dst);
    //imshow("warp",warped);
    searchForLanes(warped);
    //imshow ("test",combined);
   // imshow("original",res);
    waitKey(0);
    return 0;
}
