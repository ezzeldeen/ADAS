import cv2
import numpy as np
import findingLanes as fl
import drawLanes as dr
import time 
start_time = time.time()

def filter_color(img):
    yellow_min = np.array([15, 100, 120], np.uint8)
    yellow_max = np.array([255, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
    white_min = np.array([0, 0, 150], np.uint8)
    white_max = np.array([255, 30, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1
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
    #cv2.imshow("warped",cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR))
    return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def unwarp_image (img , src ,dst):
     M = cv2.getPerspectiveTransform(dst,src)
     #cv2.imshow("unwarped",cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR))
     return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def slope(line):
    return (float(line[3]) - line[1]) / (float(line[2]) - line[0])


def weighted_img(img, initial_img, alpha=1., bita=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img,bita,gamma)

cap = cv2.VideoCapture("project_video.mp4")

while(True):
    
    ret, frame = cap.read()    
    # convert image to HSV
    # frame = cv2.imread("images/undistorted.jpg")    
    height , width = frame.shape[:2]
    res = cv2.resize(frame,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(res , cv2.COLOR_RGB2HSV)
    
    #filter image 
    sobelx = cv2.Sobel(res[:, :, 2], cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(res[:, :, 2], cv2.CV_64F, 0, 1, ksize=15)
    dirc = dir_threshold(sobelx, sobely)
    mag = magnitude_threshold (sobelx ,sobely)
    color = filter_color(image)
    combined = np.zeros_like(dirc)
    combined[(( color == 1) & (( mag == 1 ) | ( dirc == 1)))] = 1
    #cv2.imshow("combined",combined)
    #perspective transform
    src = np.float32([[290, 230],
                [350, 230],
                [520, 340],
                [130, 340]])
    dst = np.float32([[130, 0],
                [520, 0],
                [520, 310],
                [130, 310]])
    
    warped = warp_image(combined,src,dst)
    unwarped = unwarp_image(warped,src,dst)
    lx,lxf,rx,rxf =fl.search_for_lane(warped)
    o = dr.draw(res,lx,lxf,rx,rxf,src,dst)
    #out.write(o)
    #cv2.imshow('original',res)
    cv2.imshow('output',o)
    print("--- %s seconds ---" % (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    #out.release()    
cap.release()
cv2.waitKey(0);
cv2.destroyAllWindows()

    
