import numpy as np
import cv2
import glob

# function parameter : 

#camPosition (string) = left or right
#cameraNo : 0 or 1 or 2 
def cameraCalibration(cameraNo,camPosition):    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(cameraNo)
    if not cap.isOpened():
        print ("Video device or file couldn't be opened")
        exit()

    saveImgName = camPosition #string : left or right
    saveImgNameVar=0
    while(cap.isOpened()):
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            saveImgNameVar +=1
            cv2.imwrite(saveImgName+str(saveImgNameVar)+'.png',img)
            cv2.drawChessboardCorners(img, (7,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
        
    ret, cameraMatrix, distortionCooeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print ("cameraMat:", cameraMatrix)
    print ("dist :", distortionCooeff)
    cv2.destroyAllWindows()
    return cameraMatrix , distortionCooeff , rvecs, tvecs , objpoints , imgpoints , gray.shape[::-1]