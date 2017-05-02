import cv2
import numpy as np
#import matplotlib.pyplot as plt 
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
    
def getLineEq (line):
    value =line[0,3] - line[0,1] 
    m = value / (line[0,2] - line[0,0])
    c = line[0,3] - (m*line[0,2]) 
    return m,c

def drawLane (leftLine, rightLine,img) :
    leftLineSlope , c1 = getLineEq(leftLine)
    upperLeftX = int (300 - c1 / leftLineSlope)
    downLeftX = int (100 - c1 /leftLineSlope)
    rightLineSlope , c2 = getLineEq(rightLine)
    upperRightX = int (380 - c2 / rightLineSlope)
    downRightX = int (100 - c2 /rightLineSlope)
    cv2.line(img,(upperRightX,380),(downRightX,100),(0,0,0),2)   
    #cv2.line(img,(rightLine[0,0],rightLine[0,1]),(rightLine[0,2],rightLine[0,3]),(0,0,0),2)   

    #a3 = np.array( [[ [left[0,0],left[0,1]],[newLeftX,newLeftY],[right[0,0],right[0,1]],[newRightX,newRightY]]], dtype=np.int32 )
    #cv2.fillPoly(img,a3, 255 )
    return
    
frame = 0
while (frame < 33 ) :
    img = cv2.imread(str(frame)+'.png',1)
    frame += 1
    edges=cv2.Canny(img,10,300)
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array( [[
                    [2*width/3, 3*height/8],
                    [width/3, 3*height/8],
                    [40, height],
                    [width - 40, height]]], dtype=np.int32 )
    
    regionInterestImage = region_of_interest(edges, vertices) 
    lines = cv2.HoughLinesP(regionInterestImage,1,np.pi/180,80,30,30)
    verticalLine = lines[lines[:,:,1] != lines[:,:,3]]
    least = verticalLine[:,0].min(0)
    biggest = verticalLine[:,0].max(0) 
    left = lines[lines[:,:,0] == least]
    right = lines [lines[:,:,0] == biggest]
    #for x1,y1,x2,y2 in lines[0] :
    #cv2.line(img,(left[0,0],left[0,1]),(left[0,2],left[0,3]),(0,0,0),2)   
    #cv2.line(img,(right[0,0],right[0,1]),(right[0,2],right[0,3]),(0,0,0),2)    
   # plt.fill_between(lines[0,0],lines[0,1], color='grey', alpha='0.5')
    drawLane(left,right,img)
    cv2.imshow("a",img)
    cv2.waitKey(1)
    time.sleep(0.5)
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))
