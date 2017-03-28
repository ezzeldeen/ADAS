import numpy as np 
import cv2
from matplotlib import pyplot as plt


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv2.imread('calib_radial.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    trial = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
#cv2.destroyAllWindows()


"""
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
"""