import numpy as np 
import cv2
from matplotlib import pyplot as plt
imgR = cv2.imread('1.jpg',0)
imgL = cv2.imread('2.jpg',0)
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=21)
disparity = stereo.compute(imgL,imgR)

cv2.namedWindow('1', cv2.WINDOW_NORMAL) #show normal window
cv2.imshow('1.jpg', imgR)

plt.imshow(disparity,'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()