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

def drawLane (Lines,img) :
    for x1,y1,x2,y2 in lines[0] :
        if y1 != y2 :
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
    return
    
frame = 0
while (frame < 33 ) :
    img = cv2.imread("images/"+str(frame)+'.png',1)
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
    drawLane(lines,img)
    cv2.imshow("a",img)
    cv2.waitKey(1)
    #time.sleep(0.5)
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))
