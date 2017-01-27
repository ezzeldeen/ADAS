import vrep
import sys
import time
import cv2 
import numpy as np

#close all previous connections
vrep.simxFinish(-1)

#start new connection
ClientID = vrep.simxStart('127.0.0.1',19999,True,True,500,5)

#check connection
if ClientID == -1 :
    print "Couldn't connect to vrep"
    sys.exit
    
else:
    print "Connected to vrep"
    
#motors handle
errorCode,leftMotorHandle=vrep.simxGetObjectHandle(ClientID,'leftMotor',vrep.simx_opmode_blocking)
errorCode,rightMotorHandle=vrep.simxGetObjectHandle(ClientID,'rightMotor',vrep.simx_opmode_blocking)

#camera handle
errorCode,frontCamHandle=vrep.simxGetObjectHandle(ClientID,'frontCam',vrep.simx_opmode_oneshot_wait)

#first call using streaming
errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,1,vrep.simx_opmode_streaming)

#while there is an error try to capture frame again untill errorCode become zero  
while errorCode==1:
    errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,1,vrep.simx_opmode_streaming)

#matrix of zeros to write the output frames on it
out  = np.zeros((128 ,128,3), np.uint8)

#main code    
while 1:
    #rotate image as vrep returns rotated images
    imgRotated= np.array(image,dtype=np.uint8)
    imgRotated.resize([resolution[0],resolution[1],1])
    rotation = cv2.getRotationMatrix2D((64,64),180,1.0)    
    img = cv2.warpAffine(imgRotated,rotation,(128,128))
    
    #edge detection
    #parameters are image & threshold 1 & threshold 2
    edges=cv2.Canny(img,10,300)
    
    #hough line transform
    #parameters are matrix of edges & rho "one is default value" & theta "np.pi/180 is also default value" & threshold
    lines = cv2.HoughLines(edges,1,np.pi/180,20)
    
    #convert from hough space to cartesian space then draw lines        
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(out,(x1,y1),(x2,y2),(205,100,0),2)
    #show image    
    cv2.imshow("vrepImage",out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #get new frame    
    errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,1,vrep.simx_opmode_buffer)
    #keep moving car forward    
    errorCode=vrep.simxSetJointTargetVelocity(ClientID,leftMotorHandle,0.2,vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetVelocity(ClientID,rightMotorHandle,0.2,vrep.simx_opmode_streaming)
cv2.cv.DestroyAllWindows()
    