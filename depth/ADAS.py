from cameraCalibration import cameraCalibration 
from stereoCalibrate import	stereoCalibration
import numpy as np
#import xlwt


cameraMatrix1 , distortionCooeff1 , rvecs1, tvecs1 , objpoints1 , imgpoints1, shape = cameraCalibration('rightCamImg/')
cameraMatrix2 , distortionCooeff2 , rvecs2, tvecs2 , objpoints2 , imgpoints2, shape = cameraCalibration('leftCamImg/')
#(objpoints, imgpoints1, imgpoints2,   camera_matrix1, dist_coefs1, camera_matrix2, dist_coefs2,shape)
Q = stereoCalibration(objpoints1, imgpoints1, imgpoints2,cameraMatrix1 ,distortionCooeff1, cameraMatrix2, distortionCooeff2, shape)
print ("Q :",Q)	
