import numpy as np
import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cal_mtx as Mtrans
from tqdm import tqdm

def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    '''
    Calculate details of curvature and vehicle position
    Attributes:
        Input:
            - binary_warped: warped binary image
            - left_fit: left lane line curve
            - right_fit: right lane line curve
        Output:
            - curvature
            - distance_from_center
    '''
    # Define y-value where we want radius of curvature
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 14/1080 # meters per pixel in y dimension
    xm_per_pix = 3.5/500 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = ((left_curverad + right_curverad) / 2)
    #print(curvature)
    lane_width = np.absolute(leftx[1079] - rightx[1079])
    lane_xm_per_pix = 3.5 / lane_width
    veh_pos = (((leftx[1079] + rightx[1079]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = cen_pos - veh_pos
    return curvature,distance_from_center


def draw_values(img,curvature,distance_from_center):
    '''
    Draw details of lane geometry data
    Attributes:
        Input:
            - img: RGB images
            - curvature: calculated lane curvature 
            - distance_from_center: calculated distance from lane center
        Output:
            - img
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    radius_text = "Radius of curvature: %sm"%(round(curvature//100*100))
        
    cv2.putText(img,radius_text,(100,100), font, 2,(0,0,255),2)
    # center_text = "Vehicle is %.3fm %s of the center"%(abs(distance_from_center),pos_flag)
    # cv2.putText(img,center_text,(100,150), font, 2,(0,0,255),2)
    return img

def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
    '''
    Draw lane area
    Attributes:
        Input:
            - undist: undist images
            - binary_warped: warped binary image 
            - left_fit: left lane line curve
            - right_fit: right lane line curve
        Output:
            - result
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (240,0, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result