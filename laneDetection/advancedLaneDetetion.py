import cv2
import numpy as np
import findingLanes as fl
import drawLanes as dr


def filter_color(img):
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1
    filtered = img
    filtered[((yellow_mask == 0) & (white_mask == 0))] = 0
    return binary_output

def dir_threshold (sobelx ,sobely):
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    angles = np.arctan2(abs_sobelx,abs_sobely)
    sbinary = np.zeros_like(angles)
    sbinary[(angles > 0.7) & (angles < 1.2)] = 1
    return sbinary

def magnitude_threshold (sobelx ,sobely):
    gradientMag = np.sqrt(sobelx**2 + sobely**2)
    factor = np.max(gradientMag)/255
    gradientMag = (gradientMag/factor).astype(np.uint8)
    binary_out = np.zeros_like(gradientMag)
    binary_out [(gradientMag>50) & (gradientMag<255)] =1    
    return binary_out
    
def warp_image (img , src , dst ):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def unwarp_image (img , src ,dst):
     M = cv2.getPerspectiveTransform(dst,src)
     return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def slope(line):
    return (float(line[3]) - line[1]) / (float(line[2]) - line[0])


def weighted_img(img, initial_img, alpha=1., bita=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img,bita,gamma)

img = cv2.imread('images/undistorted.jpg',1)

#convert image to HSV
image = cv2.cvtColor(img , cv2.COLOR_RGB2HSV)

#apply gaussian blur to remove noise
image = cv2.GaussianBlur(image,(3,3),0)

#filter image 
sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=15)
sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=15)
dirc = dir_threshold(sobelx, sobely)
mag = magnitude_threshold (sobelx ,sobely)
color = filter_color(image)
combined = np.zeros_like(dirc)
combined[(( color == 1) & (( mag == 1 ) | ( dirc == 1)))] = 1

#perspective transform
src = np.float32([[580, 460],
            [700, 460],
            [1040, 680],
            [260, 680]])
dst = np.float32([ [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720]])
warped = warp_image(combined,src,dst)
unwarped = unwarp_image(warped,src,dst)
lx,lxf,rx,rxf =fl.search_for_lane(warped)
o = dr.draw(img,lx,lxf,rx,rxf,src,dst)
cv2.imshow("out",o)
#cv2.imshow("rr",unwarped)
cv2.waitKey(0)
cv2.destroyAllWindows()


