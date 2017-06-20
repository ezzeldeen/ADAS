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

Mat unwarp_image(Mat img, Point2f* src_vertices , Point2f* dst_vertices)
{
    Mat M,warped;
    M = getPerspectiveTransform(dst_vertices,src_vertices);
    warpPerspective(img,warped, M,warped.size());
    return warped;

}

Mat polyFit(Mat &points ,int degree)
{
    int numOfpts = points.rows, i=0, j=1;
    Mat x_vals(numOfpts, degree+1,CV_32FC1) , y_vals(numOfpts,1,CV_32FC1) ;
    for(i  ;i< numOfpts ; i++)
    {
        y_vals.at<float>(i,0,0) = points.at<Point>(i).y;
        x_vals.at<float>(i,0,0)= 1;
        for(j ;j < degree + 1 ;j++)
        {
            x_vals.at<float>(i,j,0) = pow(int(points.at<Point>(i).x),j) ;

        }
        j=1;
    }
    Mat x_transposed,first_term;
    transpose(x_vals,x_transposed);
    first_term = x_transposed * x_vals ;
    invert(first_term,first_term,DECOMP_LU);
    Mat second_term;
    second_term = first_term * x_transposed;
    Mat result =  second_term * y_vals;
    return result ;

}

void drawLane(Mat left , Mat right, Mat img, Point2f* src_vertices , Point2f* dst_vertices)
{
    vector<cv::Point> pt;
    float l=0 ,r=0 ;
    int i = img.rows-1;
    for (i ; i> 0;i--)
    {
        l = left.at<float>(2) * pow(i,2) + left.at<float>(1) * i + left.at<float>(0) ;
        r = right.at<float>(2) * pow(i,2) + right.at<float>(1) * i + right.at<float>(0) ;
        pt.push_back((Point(l,i)));
        pt.push_back((Point(r,i)));
    }
    const cv::Point *pts = (const cv::Point*) Mat(pt).data;
	int num_pts = Mat(pt).rows;
	Mat colored;
	colored = Mat::zeros(img.rows, img.cols, CV_32FC3);
	Mat unwarped,out;
	fillPoly(colored, &pts,&num_pts, 1,Scalar(0,255,0),8);
	unwarped = unwarp_image(colored,src_vertices,dst_vertices);
	//addWeighted(img,1, unwarped, 0.3, 0, out);
    imshow("yaaaaaaaaaarb",unwarped);
}

void searchForLanes(Mat img, Point2f* src_vertices , Point2f* dst_vertices,Mat orignal_img)
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
    Mat leftLane;
    Mat rightLane;
    Mat current_nonZero;
    int win_y_low, win_y_high, win_left_x_low, win_left_x_high, win_right_x_low, win_right_x_high;
    int flag=0;
    for(int i = 0 ; i < numOfWindows; i++)
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
            add(current_nonZero,Scalar(win_y_low,win_left_x_low),current_nonZero);
            if(!flag)
            {
                leftLane = current_nonZero;
                flag++;
            }
            else
            {
                vconcat(leftLane, current_nonZero, leftLane);
            }
        }
        if(countNonZero(img(Range(win_y_low,win_y_high),Range(win_right_x_low,win_right_x_high))))
        {
            findNonZero(img(Range(win_y_low,win_y_high),Range(win_right_x_low,win_right_x_high)),current_nonZero);
            add(current_nonZero,Scalar(win_y_low,win_right_x_low),current_nonZero);
            if(flag == 1)
            {
                rightLane = current_nonZero;

            }
            else
            {
                vconcat(rightLane, current_nonZero, rightLane);
            }
        }

    }
    Mat left_fit = polyFit(leftLane,2);
    Mat right_fit = polyFit(rightLane,2);
    drawLane(left_fit , right_fit, orignal_img,src_vertices,dst_vertices);

}






int main()
{
    Mat img,res,outColor,grad_x,grad_y,outDir,outMag,combined,warped,HSV_img;
    img = imread("undistorted.jpg");
    resize(img, res, Size(img.cols * 0.5,img.rows * 0.5), 0, 0, CV_INTER_CUBIC);
    cvtColor(res, HSV_img , CV_RGB2HSV);
    outColor = filter_color(HSV_img);
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
    searchForLanes(warped,src,dst,res);
    //imshow ("test",combined);
   // imshow("original",res);
    waitKey(0);
    return 0;
}
