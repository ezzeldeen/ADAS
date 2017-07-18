import cv2
import numpy as np
import findingLanes as fl
import drawLanes as dr
import time 
start_time = time.time()

#def filter_color(img):
#yellow_min = np.array([0, 0, 175], np.uint8)
#yellow_max = np.array([255, 255, 25], np.uint8)
#binary_output = np.zeros_like(img[:, :, 0])
#binary_output[(yellow_mask != 0)] = 1
#return yellow_mask

def filter_color(img):
        ret,thresh_yellow_ch2 = cv2.threshold(img[:,:,1],100,255,cv2.THRESH_BINARY)
        ret,thresh_yellow_ch3 = cv2.threshold(img[:,:,2],120,255,cv2.THRESH_BINARY)
        ret,thresh_white_ch1 = cv2.threshold(img[:,:,0],0,255,cv2.THRESH_BINARY)
        ret,thresh_white_ch2 = cv2.threshold(img[:,:,1],0,30,cv2.THRESH_BINARY)
        mask = (thresh_yellow_ch2 |  thresh_yellow_ch3)  & (thresh_white_ch1 |thresh_white_ch2)
        return mask

    
def warp_image (img , src , dst ):
    M = cv2.getPerspectiveTransform(src, dst)
 #   cv2.imshow("warped",cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR))
    return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
def unwarp_image (img , src ,dst):
     M = cv2.getPerspectiveTransform(dst,src)
     #cv2.imshow("unwarped",cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR))
     return cv2.warpPerspective(img ,M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    

def weighted_img(img, initial_img, alpha=1., bita=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img,bita,gamma)
    
def measure_curvature(left_fit,right_fit,leftx,rightx,lefty,righty, img):
        ploty = np.linspace(0, img.shape[0] -1 , num=img.shape[0])  
        y_eval = np.max(ploty)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        print left_curverad        
        lane_leftx = left_fit[0] * (img.shape[0] - 1) ** 2 + left_fit[1] * (img.shape[0] - 1) + left_fit[2]
        lane_rightx = right_fit[0] * (img.shape[0] - 1) ** 2 + right_fit[1] * (img.shape[0] - 1) + right_fit[2]
        car_pos = ((img.shape[1] / 2) - ((lane_leftx + lane_rightx) / 2)) * xm_per_pix
        return (left_curverad + right_curverad) / 2, car_pos.round(2)
        
cap = cv2.VideoCapture("project_video.mp4")
out = cv2.VideoWriter('output.avi',-1, 20.0, (1280,720))

f = 1
while(True):
    ret, frame = cap.read()    
#frame = cv2.imread("images/undistorted.jpg")
    height , width = frame.shape[:2]
    res = cv2.resize(frame,(width, height), interpolation = cv2.INTER_CUBIC)
    r = filter_color(res)
    #cv2.imshow("fF",r)
    
    src = np.float32([[580, 460],
            [700, 460],
            [1040, 680],
            [260, 680]])
    dst = np.float32([[260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720]])
    
    warped = warp_image(r,src,dst)
    lx,lxf,rx,rxf,ly,ry =fl.search_for_lane(warped)
    o = dr.draw(res,lx,lxf,rx,rxf,src,dst)
    ROC , carPosition = measure_curvature(lxf,rxf,lx,rx,ly,ry,res)
    if carPosition > 0:
        car_pos_text = '{}m right of center'.format(carPosition)
    else:
        car_pos_text = '{}m left of center'.format(abs(carPosition))

    #cv2.putText(o, "Lane curve: {}m".format(ROC.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #            color=(255, 255, 255), thickness=2)
    cv2.putText(o, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)
    #cv2.imshow('original',war+ped)
    cv2.imshow('output',o)
    out.write(o)
    print("--- %s seconds ---" % (time.time() - start_time))
    print f
    f +=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
     
out.release()    
cap.release()
cv2.waitKey(0);
cv2.destroyAllWindows()