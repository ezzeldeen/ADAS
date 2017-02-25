
#import vrep
import sys
import numpy as np
import timeit
import math
import sys

maxRadarDist= 15
deltaX=maxRadarDist/3   # at point B
safeDistance=(maxRadarDist/3)*2 # at point A
desiredSpeed=50/float(3)
car1Speed=desiredSpeed
detectedState=True
detectedDistance=130
setSpeed=desiredSpeed #initialy, current speed= desiredSpeed
time1 = 0
# point A : leading car, point C: myCar, point B: deltaX
start = timeit.default_timer()


acceleration=0
counter=0
car2Speed = -1 # initialy I don't know
print ("dis     : ",detectedDistance)
print ("car1Vel : ",car1Speed)
print ("car2Vel : ",car2Speed)
while car2Speed<car1Speed:
    #TODO: each iteration update detectedState & detectedDistance
    #TODO: Check if it is the same object or has been changed
    time1 +=0.5
    distBetweenCars = None
    print ("time    : ", time1)
    if detectedState==False:
        counter=0   #do nothing
    else:
        counter +=1
        #calc. dist
        if counter ==1:
            d1=detectedDistance
            print ("dis     : ", d1)
            #time1 = 0                    #timeit.default_timer() #real time
        #calc. velocity of car2
        elif counter==2:
            #TODO read d2 from RADAR not by calculations
            d2 = detectedDistance-5
            print ("dis     : ", d2)
            car2Speed=(d2-d1)/float(time1)+car1Speed #final velocity (car2 velocity)
            if (car2Speed<car1Speed):
                # calc. acceleration
                acceleration = (float)(math.pow(car2Speed,2) - math.pow(car1Speed,2)) / (2*(d2))
                car1Speed =car1Speed + (acceleration*time1)
        elif counter>=3:
            #to calculate the dis between the 2 car after 2sec
            #####
            disCar1 = (time1*car1Speed) + (0.5*acceleration*math.pow(time1,2))
            disCar2 = (time1 * car2Speed) + (0.5 * acceleration * math.pow(time1, 2))
            distBetweenCars=disCar2+detectedDistance-disCar1
            #####
            car1Speed = car1Speed + (acceleration * time1)
            print ("dis     : ", distBetweenCars)
    print ("car1Vel : ", car1Speed)
    print ("car2Vel : ", car2Speed)
    print ("    ")

"""
    ### CASE 1 : myCar bet. point B and C
    if (detectedDistance > deltaX && detectedDistance < maxRadarDist):
        s=maxRadarDist-detectedDistance
        setSpeed=(setSpeed^2)-((2*a*s)^0.5)
"""



