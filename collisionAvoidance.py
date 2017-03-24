# -*- coding: utf-8 -*-
#Import Libraries:
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting

#Pre-Allocation

PI=math.pi  #pi=3.14..., constant

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print 'Connected to remote API server'
    
else:
    print 'Connection not successful'
    sys.exit('Could not connect')


#retrieve motor  handles
errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)


sensor_h=[] #empty list for handles
sensor_val=np.array([]) #empty array for sensor measurements

#orientation of all the sensors: 
sensor_loc=np.array([-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,-170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 

#for loop to retrieve sensor arrays and initiate sensors
for x in range(1,16+1):
        errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
        sensor_h.append(sensor_handle) #keep list of handles        
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)                
        sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
        

t = time.time()


while (time.time()-t)<60:
    #Loop Execution
    sensor_val=np.array([])    
    for x in range(1,16+1):
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)                
        sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values

    
    #controller specific
    sensor_sq=sensor_val[0:8]*sensor_val[0:8] #square the values of front-facing sensors 1-8
        
    min_ind=np.where(sensor_sq==np.min(sensor_sq))
    min_ind=min_ind[0][0]
    
    if sensor_sq[min_ind]<0.2:
        steer=-1/sensor_loc[min_ind]
    else:
        steer=0
            
    
    v=1	#forward velocity
    kp=0.5	#steering gain
    vl=v+kp*steer
    vr=v-kp*steer
    print "V_l =",vl
    print "V_r =",vr

    errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)


    time.sleep(0.2) #loop executes once every 0.2 seconds (= 5 Hz)

#Post ALlocation
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
    