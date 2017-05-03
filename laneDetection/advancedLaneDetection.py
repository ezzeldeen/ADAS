import cv2
from matplotlib import pyplot as plt
import numpy as np

def drawLane (leftLine, rightLine,img) :
    cv2.line(img,(leftLine[0,0],leftLine[0,1]),(leftLine[0,2],leftLine[0,3]),(0,0,0),2)    
    cv2.line(img,(rightLine[0,0],rightLine[0,1]),(rightLine[0,2],rightLine[0,3]),(0,0,0),2)    
    return
    
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


img = cv2.imread('images/udacity.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray_image,(3,3),0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
height = img.shape[0]
width = img.shape[1]

pts = np.array( [[  [2*width/3, 3*height/8],
                    [width/3, 3*height/8],
                    [40, height-150],
                    [width - 40, height-150]]], dtype=np.float32 )
                    
vertices = np.array( [[
                     [2*width/3, 3*height/8],
                    [width/3, 3*height/8],
                    [40, height-100],
                    [width - 40, height-100]]], dtype=np.int32 )                    

regionInterestImage = region_of_interest(img, vertices) 

dst = np.array([[width, 0],
                    [0, 0], [0, height],
                    [width ,height]],dtype="float32")

transform = cv2.getPerspectiveTransform(pts,dst) 
dsta = cv2.warpPerspective(img,transform,(width,height))
lines = cv2.HoughLinesP(regionInterestImage,1,np.pi/180,80,30,30)
verticalLine = lines[lines[:,:,1] != lines[:,:,3]]
least = verticalLine[:,0].min(0)
biggest = verticalLine[:,0].max(0) 
left = lines[lines[:,:,0] == least]
right = lines [lines[:,:,0] == biggest]
drawLane(left,right,img)
drawLane(left,right,dsta)


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(regionInterestImage,cmap = 'gray')
plt.title('pts'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(dsta,cmap = 'gray')
plt.title('transformed'), plt.xticks([]), plt.yticks([])

plt.show