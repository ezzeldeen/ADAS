import cv2
import numpy as np

def search_for_lane(img):
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
        cv2.rectangle(img,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high) ,(0,0,255),2)
        cv2.rectangle(img,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high) ,(0,0,255),2)
        
    