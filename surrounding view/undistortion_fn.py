import numpy as np
import cv2

camera_matrix = np.array([[ 735.8018328 , 0.0 , 984.2469258 ],
               [ 0.0 , 746.98821085 , 495.21819959 ],
               [ 0.0 , 0.0 , 1.0 ]]);
               
distortion_coefficient = np.array([ -0.487127 , 0.27952148 , 0.00717382 , -0.00374102 , -0.06300308 ]);

img = cv2.imread('distorted/1.jpg')
dst = cv2.undistort(img, camera_matrix, distortion_coefficient, None)
cv2.imshow('pic',dst)
cv2.waitKey(0)
#cv2.imwrite('.png',dst)

#cap = cv2.VideoCapture(0)
#cap.set(3,640);
#cap.set(4,480);

#while(True):
#
#        ret, frame = cap.read()
#
#        gray = cv2.undistort(frame, camera_matrix, distortion_coefficient, None)
#
#        cv2.imshow('frame',gray)
#        if  0xFF == ord('q'):
#            break

#cap.release()
cv2.destroyAllWindows()