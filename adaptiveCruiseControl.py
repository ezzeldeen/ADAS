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

errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)

# to make it steering 
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,4,vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,4,vrep.simx_opmode_streaming)
#errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,1.5,vrep.simx_opmode_streaming)
s = 2
print s
