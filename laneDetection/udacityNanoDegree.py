import cv2
import numpy as np

PREV_LEFT_X1 = None
PREV_LEFT_X2 = None
PREV_RIGHT_X1 = None
PREV_RIGHT_X2 = None

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
    

def slope(line):
    return (float(line[3]) - line[1]) / (float(line[2]) - line[0])


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    global PREV_LEFT_X1, PREV_LEFT_X2, PREV_RIGHT_X1, PREV_RIGHT_X2
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for line in lines:
        line = line[0]
        s = slope(line)

        if 0.3 > s > -0.3:
            continue

        if s < 0:
            if line[0] > img.shape[1] / 2 + 40:
                continue

            left_x += [line[0], line[2]]
            left_y += [line[1], line[3]]
            # cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [0, 0, 255], thickness)
        else:
            if line[0] < img.shape[1] / 2 - 40:
                continue

            right_x += [line[0], line[2]]
            right_y += [line[1], line[3]]
            # cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [255, 255, 0], thickness)

    y1 = img.shape[0]
    y2 = img.shape[0] / 2 + 90

    if len(left_x) <= 1 or len(right_x) <= 1:
        if PREV_LEFT_X1 is not None:
            cv2.line(img, (int(PREV_LEFT_X1), int(y1)), (int(PREV_LEFT_X2), int(y2)), color, thickness)
            cv2.line(img, (int(PREV_LEFT_X2), int(y1)), (int(PREV_RIGHT_X2), int(y2)), color, thickness)
        return

    left_poly = np.polynomial.Polynomial.fit(np.array(left_x), np.array(left_y), 1)
    right_poly = np.polynomial.Polynomial.fit(np.array(right_x), np.array(right_y), 1)

    left_x1 = (left_poly - y1).roots()
    right_x1 = (right_poly - y1).roots()

    left_x2 = (left_poly - y2).roots()
    right_x2 = (right_poly - y2).roots()

    if PREV_LEFT_X1 is not None:
        left_x1 = PREV_LEFT_X1 * 0.7 + left_x1 * 0.3
        left_x2 = PREV_LEFT_X2 * 0.7 + left_x2 * 0.3
        right_x1 = PREV_RIGHT_X1 * 0.7 + right_x1 * 0.3
        right_x2 = PREV_RIGHT_X2 * 0.7 + right_x2 * 0.3

    PREV_LEFT_X1 = left_x1
    PREV_LEFT_X2 = left_x2
    PREV_RIGHT_X1 = right_x1
    PREV_RIGHT_X2 = right_x2

    cv2.line(img, (int(left_x1), int(y1)), (int(left_x2), int(y2)), color, thickness)
    cv2.line(img, (int(right_x1), int(y1)), (int(right_x2), int(y2)), color, thickness)    

def weighted_img(img, initial_img, alpha=1., bita=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img,bita,gamma)

img = cv2.imread('images/carLane.jpg',1)

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



#apply canny edge detection
image = cv2.Canny(image,30,130)

#region of interest
height,width=img.shape[:2]
s = np.array([[[40,height],[width/2,height/2+40],[width/2,height/2 +40],[width-40,height]]],np.int32)
image = region_of_interest(image,s)

#run hough transform
image = cv2.HoughLinesP(image,1,np.pi/90,10,np.array([]),15,110)
line_img = np.zeros_like(image)
#draw_lines(line_img,image,thickness=7)
#out = weighted_img(image,img,bita=250.)

#src = np.array([[340,280],[430,280],[670,410],[175,410]],np.float32)
#dst = np.array([[150,0],[550,0],[550,630],[150,630]],np.float32)
#M = cv2.getPerspectiveTransform(src, dst)
#warp = cv2.warpPerspective(test.copy(), M, (800, 600))

cv2.imshow("out",combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


