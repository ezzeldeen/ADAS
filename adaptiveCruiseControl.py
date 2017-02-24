
#import vrep
import sys
import numpy as np
import timeit
import math
maxRadarDist= 15  
deltaX=maxRadarDist/3   # at point B
safeDistance=(maxRadarDist/3)*2 # at point A
desiredSpeed=50/float(3)
car1Speed=desiredSpeed
detectedState=True
detectedDistance=130
setSpeed=desiredSpeed #initialy, current speed= desiredSpeed
# point A : leading car, point C: myCar, point B: deltaX
start = timeit.default_timer()

c = 0
while 1:
    c += 1
    if c == 10000:
        break

stop = timeit.default_timer()
print stop - start


counter=0
while 1:
    #TODO: each iteration update detectedState & detectedDistance
    #TODO: Check if it is the same object or has been changed
    if detectedState==False:
        counter=0   #do nothing
    else:
        counter +=1
        #calc. dist
        if counter ==1:
            d1=detectedDistance
            time1 = timeit.default_timer()
        #calc. velocity of car2
        elif counter>=2:
            #TODO read d2 from RADAR not by calculations
            d2 = detectedDistance-5
            #DELAY
            ##############
            c = 0
            while 1:
                c += 1
                if c == 10000:
                    break
            ##############
            time2 = timeit.default_timer()
            print time2-time1
            car2Speed=(d2-d1)/float(time2-time1)+car1Speed #final velocity (car2 velocity)
            if (car2Speed<car1Speed):
                # calc. acceleration
                acceleration = (float)(math.pow(car2Speed,2) - math.pow(car1Speed,2)) / (2*(d2))


"""
    ### CASE 1 : myCar bet. point B and C
    if (detectedDistance > deltaX && detectedDistance < maxRadarDist):
        s=maxRadarDist-detectedDistance
        setSpeed=(setSpeed^2)-((2*a*s)^0.5)
"""



