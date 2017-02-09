# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 04:55:15 2017

@author: MoHassan
"""

import vrep
import sys
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')
else:
	print ('connection not successful')
	#sys.exit('could no connect')

errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'car1_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'car1_rightMotor',vrep.simx_opmode_oneshot_wait)


desiredSpeed=5
safetyDistance=1 # 1 meter

# to make it steering 
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,desiredSpeed,vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,desiredSpeed,vrep.simx_opmode_streaming)
#errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,1.5,vrep.simx_opmode_streaming)
## inifinty loop
while 1 :
    # DETERMINE distance bet. 2 cars -> call it diffDistance
    # determine speed of the second car
    ### case 1 : danger case
    # if (diffDistance < safetyDistance && car2Speed < car1speed)
    # then -> make decelerare car1
    ### case 2 : not danger
    # else if (diffDistance< safetyDistance && car2Speed > car1speed)
    # then -> set car1Speed to desired speed
    
s = 2
print s
