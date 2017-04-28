import numpy as np
import cv2
import glob
import os
import re

def cameraCalibration(path):
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  directory = path

  def sort_humanly( l ):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l
  images = sort_humanly(os.listdir(directory))


  for file_name in images:
    name =directory + file_name
    print(name)
    img = cv2.imread(name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
      objpoints.append(objp)

      corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
      imgpoints.append(corners2)

      # Draw and display the corners
      img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
      cv2.imshow('img',img)
      cv2.waitKey(500)

  ret, cameraMatrix, distortionCooeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
  cv2.destroyAllWindows()
  return cameraMatrix,distortionCooeff, rvecs,tvecs,objpoints,imgpoints,gray.shape[::-1]


#cameraMatrix2 , distortionCooeff2 , rvecs2, tvecs2 , objpoints2 , imgpoints1, shape = cameraCalibration('leftCamImg/')