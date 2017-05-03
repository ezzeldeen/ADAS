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

def filter_color(img):
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)
    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(yellow_mask, white_mask))
    return img

    
img = cv2.imread('images/carLane.jpg',1)

#convert image to HSV
image = cv2.cvtColor(img , cv2.COLOR_RGB2HSV)
#apply gaussian blur to remove noise
image = cv2.GaussianBlur(image,(3,3),0)
#filter image 
image = filter_color(image)
#apply canny edge detection
image = cv2.Canny(image,30,130)
#region of interest
height,width=img.shape[:2]
s = np.array([[[40,height],[width/2,height/2+40],[width/2,height/2 +40],[width-40,height]]],np.int32)
image = region_of_interest(image,s)

#sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
#sobelx = sobelx.astype(np.uint8)
#ret,th1 = cv2.threshold(sobelx,30,150,cv2.THRESH_BINARY)

#second path
#HLS_img = cv2.cvtColor(img , cv2.COLOR_RGB2HLS)
#s_channel = HLS_img[:,:,1]
#mask = cv2.inRange(s_channel,130,270)
#masked_image = cv2.bitwise_and(th1, mask)
#src = np.array([[340,280],[430,280],[670,410],[175,410]],np.float32)
#ss = np.array([[[340,280],[430,280],[670,410],[175,410]]],np.int32)
#dst = np.array([[150,0],[550,0],[550,630],[150,630]],np.float32)
#test = region_of_interest(masked_image,ss)
#M = cv2.getPerspectiveTransform(src, dst)
#warp = cv2.warpPerspective(test.copy(), M, (800, 600))

cv2.imshow("masked",image)
#cv2.imshow("original",masked_image)
#cv2.imshow("warp",warp)
cv2.waitKey(0)
cv2.destroyAllWindows()


