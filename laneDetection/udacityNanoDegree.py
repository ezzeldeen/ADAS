import cv2
import numpy as np

def region_of_interest(img, vertices):    
    #defining a blank mask to start with
    mask = np.zeros_like(img)       
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

img = cv2.imread('images/udacity.png',1)

#first path
gray_img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
sobelx = sobelx.astype(np.uint8)
ret,th1 = cv2.threshold(sobelx,30,150,cv2.THRESH_BINARY)

#second path
HLS_img = cv2.cvtColor(img , cv2.COLOR_RGB2HLS)
s_channel = HLS_img[:,:,1]
mask = cv2.inRange(s_channel,130,270)

masked_image = cv2.bitwise_and(th1, mask)
src = np.array([[350,250],[100,450],[700,450],[450,250]],np.float32)
dst = np.array([[0,0],[0,600],[600,600],[600,0]],np.float32)
ss = np.array([[[350,250],[100,450],[700,450],[450,250]]],np.int32)

test = region_of_interest(masked_image,ss)
M = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(masked_image.copy(), M, (600, 600))
cv2.imshow("warp",test)
cv2.imshow("output",warp)
cv2.waitKey(0)
cv2.destroyAllWindows()


