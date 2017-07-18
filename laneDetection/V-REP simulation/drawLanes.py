import cv2
import numpy as np 
import time
def draw (img, leftx, left_fit, rightx, right_fit, src, dst , detected ):
    if (detected):
        colored_warp = np.zeros_like(img).astype(np.uint8)
        y = np.linspace(0, img.shape[0]-1, img.shape[0])
        left = left_fit[0] * y **2 + left_fit[1] * y + left_fit[2]
        right = right_fit[0] * y **2 + right_fit[1] * y + right_fit[2]
        visualize = np.zeros_like(img).astype(np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left,y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right,y])))])
        pts = np.hstack((pts_left,pts_right))
        pts = np.array(pts,dtype=np.int32)
        cv2.fillPoly(colored_warp,pts,(0,255,0))
        cv2.polylines(visualize,pts,False,(255,255,255)) 
        M = cv2.getPerspectiveTransform(dst,src)
        unwarped = cv2.warpPerspective(colored_warp ,M, (colored_warp.shape[1], colored_warp.shape[0]), flags=cv2.INTER_LINEAR)
        result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)    
        return result
    else : 
        return img
   
    		