#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>

using namespace cv ;

int main()
{
    Mat img ;
    img = imread("images.jpg");
    if(! img.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "test", CV_WINDOW_AUTOSIZE );
    imshow ("test",img);
    waitKey(0);
    return 0;
}
