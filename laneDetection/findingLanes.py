import cv2
import numpy as np

def search_for_lane(img):
    histogram = np.sum(img[int( img.shape[0] / 2 ):, :],axis =0) 
    mid_point = np.int( histogram.shape[0]/2 )
    left_point = np.argmax(histogram[:mid_point])
    right_point = np.argmax(histogram[mid_point:])
    num_of_windows = 9 
    window_height = np.int(img.shape[0] / num_of_windows)
    