import cv2
import numpy as np
import findingLanes as fl
import drawLanes as dr
import time 
start_time = time.time()
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
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow("b",masked_image)    
    return masked_image

def filter_color(img):
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
    white_min = np.array([20, 30, 80], np.uint8)
    white_max = np.array([255, 255, 255], np.uint8)
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

frame = 1
while(frame < 33 ):
    img = cv2.imread('images/'+str(frame)+'.png',1)
    frame += 1 
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
    src = np.float32([[280, 205],
                [370, 205],
                [470, 330],
                [190, 330]])
    dst = np.float32([[60, 20],
                [600, 20],
                [600, 470],
                [60, 470]])
    
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[
                    [2*width/3, 3*height/8],
                    [width/3, 3*height/8],
                    [40, height],
                    [width - 40, height]]], dtype=np.int32 )
    test= region_of_interest(img,vertices)            
    warped = warp_image(combined,src,dst)
    unwarped = unwarp_image(warped,src,dst)
    lx,lxf,rx,rxf =fl.search_for_lane(warped)
    o = dr.draw(img,lx,lxf,rx,rxf,src,dst)
    cv2.imshow("out",o)
    cv2.waitKey(1)
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))

    
