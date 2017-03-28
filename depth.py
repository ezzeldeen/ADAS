import numpy as np 
import cv2
from matplotlib import pyplot as plt
"""
img = cv2.imread('calib_radial.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.

img = cv2.imread('calib_radial.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints1.append(corners2)

    # Draw and display the corners
    trial = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
    rms, camera_matrix1, dist_coefs1, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints1, gray.shape[::-1],None,None)

T=None
R= None
print (camera_matrix1)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints1,   camera_matrix1, dist_coefs1, camera_matrix1, dist_coefs1,gray.shape[::-1], R, T, E=None, F=None, flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TermCriteria_MAX_ITER+cv2.TermCriteria_EPS, 30, 1e-6) )
print (cameraMatrix1)
print (cameraMatrix2)

print (R)
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