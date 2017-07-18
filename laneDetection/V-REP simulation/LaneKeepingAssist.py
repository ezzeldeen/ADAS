import vrep
import sys
import cv2 
import numpy as np
import findingLanes as fl
import drawLanes as dr
import winsound
Freq = 2500 # Set Frequency To 2500 Hertz
Dur = 1000 # Set Duration To 1000 ms == 1 second

def filter_color(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
       # cv2.imshow("ff",img)
        ret,thresh_yellow_ch2 = cv2.threshold(img[:,:,1],100,255,cv2.THRESH_BINARY)
        ret,thresh_yellow_ch3 = cv2.threshold(img[:,:,2],60,255,cv2.THRESH_BINARY)
        ret,thresh_white_ch1 = cv2.threshold(img[:,:,0],0,255,cv2.THRESH_BINARY)
        ret,thresh_white_ch2 = cv2.threshold(img[:,:,1],0,30,cv2.THRESH_BINARY)
        mask = (thresh_yellow_ch2 |  thresh_yellow_ch3)  & (thresh_white_ch1 |thresh_white_ch2)
        return mask
        
def warp_image (img , src , dst ):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def unwarp_image (img , src ,dst):
     M = cv2.getPerspectiveTransform(dst,src)
     return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def weighted_img(img, initial_img, alpha=1., bita=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img,bita,gamma)
    
def calc_vehicle_offset(img, left_fit, right_fit):
    # Calculate vehicle center offset in pixels
    bottom_y = img.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = img.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
    # Convert pixel offset to meters
    xm_per_pix = 4.0/90 # meters per pixel in x dimension
    vehicle_offset = abs(vehicle_offset) * xm_per_pix
    return vehicle_offset
    
src = np.float32([[100, 320],
        [430, 320],
        [500, 430],
        [40, 430]])        

dst = np.float32([[40, 0],
        [500, 0],
        [500, 512],
        [40, 512]])
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
#errorCode,leftMotorHandle=vrep.simxGetObjectHandle(ClientID,'leftMotor',vrep.simx_opmode_blocking)
#errorCode,rightMotorHandle=vrep.simxGetObjectHandle(ClientID,'rightMotor',vrep.simx_opmode_blocking)

#camera handle
errorCode,frontCamHandle=vrep.simxGetObjectHandle(ClientID,'frontCam',vrep.simx_opmode_oneshot_wait)

#get motor handles
#errorCode,leftMotorHandle=vrep.simxGetObjectHandle(ClientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
#errorCode,rightMotorHandle=vrep.simxGetObjectHandle(ClientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_blocking)

#first call using streaming
errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,0,vrep.simx_opmode_streaming)

#while there is an error try to capture frame again untill errorCode become zero  
while errorCode==1:
    errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,0,vrep.simx_opmode_streaming)


#main code    
while 1:
    #rotate image as vrep returns rotated images
    imgRotated= np.array(image,dtype=np.uint8)
    imgRotated.resize([resolution[0],resolution[1],3])
    rotation = cv2.getRotationMatrix2D((256,256),180,1.0)    
    img = cv2.warpAffine(imgRotated,rotation,(512,512))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    i = filter_color(img)
    warped = warp_image(i,src,dst)
    lx,lxf,rx,rxf,flag =fl.search_for_lane(warped)
    o = dr.draw(img,lx,lxf,rx,rxf,src,dst,flag)
    distance = calc_vehicle_offset(img,lxf,rxf)
    print distance
    if distance > 2.8 :
        winsound.Beep(Freq,Dur)

    
    #image_byte_array = array.array('b',image)
    #im = Image.frombuffer("RGB", (512,512), image_byte_array, "raw", "RGB", 0, 1)
    #im.show()
    #show image    
    cv2.imshow("vrepImage",o)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #get new frame    
    errorCode,resolution,image=vrep.simxGetVisionSensorImage(ClientID,frontCamHandle,0,vrep.simx_opmode_buffer)
    #keep moving car forward    
    #errorCode=vrep.simxSetJointTargetVelocity(ClientID,leftMotorHandle,0.2,vrep.simx_opmode_streaming)
    #errorCode=vrep.simxSetJointTargetVelocity(ClientID,rightMotorHandle,0.2,vrep.simx_opmode_streaming)
cv2.cv.DestroyAllWindows()
    