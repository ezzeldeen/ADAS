import numpy as np
import cv2

def stereoCalibration(objpoints, imgpoints1, imgpoints2,   camera_matrix1, dist_coefs1, camera_matrix2, dist_coefs2,shape):	
	T=None
	R= None
	retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints1,   camera_matrix1, dist_coefs1, camera_matrix1, dist_coefs1,shape, R, T, E=None, F=None, flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TermCriteria_MAX_ITER+cv2.TermCriteria_EPS, 30, 1e-6))
	R1=None
	R2=None
	P1=None
	P2=None
	Q=None
	R1,R2,P1,P2,Q,unKnown1,unKnown2=cv2.stereoRectify(camera_matrix1,dist_coefs1,camera_matrix2,dist_coefs2,shape,R,T,R1,R2,P1,P2,Q,flags=cv2.CALIB_ZERO_DISPARITY,alpha=1,newImageSize= None)

	return Q
