import cv2
import numpy as np

def search_for_lane(img):
    out_img = np.dstack((img, img, img)) * 255
    histogram = np.sum(img[int( img.shape[0] / 2 ):, :],axis =0) 
    mid_point = np.int( histogram.shape[0]/2 )
    left_point = np.argmax(histogram[:mid_point])
    right_point = np.argmax(histogram[mid_point:])
    num_of_windows = 9 
    window_height = np.int(img.shape[0] / num_of_windows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    current_right = right_point 
    current_left = left_point
    margin = 100 
    minpix = 50 
    left_lane = []
    right_lane = []
    for window in range(0,num_of_windows):
        win_y_low = img.shape[0] - ((window +1 ) * window_height) 
        win_y_high = win_y_low + window_height 
        win_x_left_low = current_left - margin
        win_x_left_high = current_left + margin        
        win_x_right_low = current_right -  margin        
        win_x_right_high = current_right + margin
        cv2.rectangle(out_img,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high) ,(0,255,0),2)
        cv2.rectangle(out_img,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high) ,(0,0,255),2)
        good_left_lane = (( nonzeroy >= win_y_low ) & (nonzeroy < win_y_high ) & 
                            ( nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high )).nonzero()[0]
        good_right_lane = (( nonzeroy >= win_y_low ) & (nonzeroy < win_y_high ) & 
                            ( nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high )).nonzero()[0]
        left_lane.append(good_left_lane)
        right_lane.append(good_right_lane)
        if len(good_left_lane) > minpix :
            current_left = np.int(np.mean(nonzerox[good_left_lane]))                    
        if len(good_right_lane) > minpix :
            current_right = np.int(np.mean(nonzerox[good_right_lane]))
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)
    leftx = nonzerox[left_lane]
    rightx= nonzerox[right_lane]
    lefty = nonzeroy[left_lane]
    righty = nonzeroy[right_lane]
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    return leftx,left_fit,rightx,right_fit    
        
                    
                            
    return

   
    