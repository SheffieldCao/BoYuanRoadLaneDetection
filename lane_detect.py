import numpy as np   
import os
import matplotlib.pyplot as plt
import cv2
import utils
import tqdm 
import scipy
from tqdm import tqdm
from scipy.sparse import csr_matrix

import cal_mtx as Mtrans
import filter_tools
import utils

def find_line(binary_warped):
    '''
    Find lane lines in a binary warped image
    Method Description:
        - Find the most possible start point of each lane line;
        - Using slide window to find the most possible next point of each lane line; 
        - Fit a parabola or a cubic curve by using all the discrete lane line points 
    Attitudes:
        Input: 
            - binary_warp: warped result from binary image(binaryed from filtered RGB images)
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
  
    leftx_base = np.argmax(histogram[550:750]) + 550
    rightx_base = np.argmax(histogram[1050:1250]) + 1050
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()  # (axis0,axis1) position of nonzero pixels

    nonzeroy = np.array(nonzero[0]) 
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, leftx_base, rightx_base

def detect_img(img,M,Minv,camera_params_path = 'calibrate_res/CameraParams.npz'):
    '''
    Lane detection in a single image.
    Attitudes:
        Input: 
            - img: RGB images
            - M: perspective transform matrix(from vehicle view to bird's eye view)
            - Minv: inverse perspective transform matrix of M
    '''
    camera_params = np.load(camera_params_path)
    distoration = camera_params['dist']
    cam_matrix = camera_params['mtx']
    img_undst = cv2.undistort(img, cam_matrix, distoration, None, cam_matrix)
    thresholded = filter_tools.thresholding(img_undst)
    thresed_warped = Mtrans.warper(thresholded,M)
    # find line points
    left_fit, right_fit, left_lane_inds, right_lane_inds,leftx_base,rightx_base = find_line(thresed_warped)
    cur,dst = utils.calculate_curv_and_pos(thresed_warped,left_fit, right_fit)
    # using Minv to drwa detected lane areas in vehicle view
    res = utils.draw_area(img_undst,thresed_warped,Minv,left_fit,right_fit)
    # draw lane curvature on this frame
    return utils.draw_values(res,cur,dst)
    

def detect_video(M,Minv,video_path = './data/BYR0084.mp4',save_path = 'examples/video_demo.mp4',camera_params_path = 'calibrate_res/CameraParams.npz'):
    '''
    Lane detection in a video.
    Attitudes:
        Input: 
            - M: perspective transform matrix(from vehicle view to bird's eye view)
            - Minv: inverse perspective transform matrix of M
            - video_path: path to camera video, default = './data/example.mp4'
            - save_path: path to camera video, default = './examples/video_demo.mp4'
    '''
    camera_params = np.load(camera_params_path)
    distoration = camera_params['dist']
    cam_matrix = camera_params['mtx']

    file_ext = save_path.split('.')[-1]
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if file_ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif file_ext == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    video_out = cv2.VideoWriter()
    video_out.open(save_path,fourcc,fps,size,True)

    for i in tqdm(range(int(frames_num))):
        ret, frame = video.read()
        if ret:
            frame_undst = cv2.undistort(frame, cam_matrix, distoration, None, cam_matrix)
            # mask the icon
            frame_undst[960:1060,60:210] = 20
            frame_processed = detect_img(frame_undst,M,Minv)
            video_out.write(frame_processed)
        else:
            raise FileNotFoundError
            
    print('All Finished!')
    video_out.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    M = np.loadtxt('calibrate_res/M.txt')
    Minv = np.loadtxt('calibrate_res/Minv.txt')

    img = cv2.imread('data/frame_599.jpg')
    IMG = detect_img(img,M,Minv)
    cv2.imwrite('./examples/demo_frame_599.png',IMG)
