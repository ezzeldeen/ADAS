import cv2
import numpy as np 
import advancedLaneDetetion as f

def draw (img, leftx, left_fit, rightx, right_fit, src, dst ):
    
    colored_warp = np.zeros_like(img).astype(np.uint8)
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    left = left_fit[0] * y **2 + left_fit[1] * y + left_fit[2]
    right = right_fit[0] * y **2 + right_fit[1] * y + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left,y]))])
    pts_right = np.array([np.flipud(np.vstack([right,y]))])
    pts = np.hstack((pts_left,pts_right))
    pts = np.array(pts,dtype=np.int32)
    cv2.fillPoly(colored_warp,pts,(0,255,0))
    unwarped =  f.unwarp_image(colored_warp,src,dst)
    return unwarped
    