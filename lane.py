import numpy as np   
import os
import matplotlib.pyplot as plt
import cv2
import utils
import tqdm 
import cal_mtx as Mtrans
import scipy

from tqdm import tqdm
from scipy.sparse import csr_matrix

def find_line(binary_warped):
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
    nonzero = binary_warped.nonzero()  # (axis0,axis1) 坐标对，表征非零元素位置

    nonzeroy = np.array(nonzero[0]) #0维度的非零元素axis0坐标
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

def choose_linebase():
    '''测试图片的二值化，滤波与hist图，选择车道线底部起点阀值'''
    save_path = os.path.join('perp_transform_img','__day_test_Output')
    # test_dir = 'thres_test_imgs/day'
    # init_imgs = utils.get_images_by_dir(test_dir)
    for i in tqdm(range(10)):
        img = cv2.imread(os.path.join(save_path,'warped_day_test%d.jpg'%i))

        #cv2.imwrite(os.path.join(save_path,'warped_day_test%d.jpg'%i),img*255)  #cv2仅显示255灰度的图像，0-1二值图像要转化一下
        histogram1 = np.sum(img[:,300:1400]/255, axis=0)
        histogram2 = np.sum(img[img.shape[0]//2:,300:1400]/255, axis=0)
        fig,ax = plt.subplots(2,2, figsize=(18,12))
        ax[0,0].imshow(img)
        ax[1,0].plot(histogram1)
        ax[1,0].set_xticks([0,300,600,900,1200])
        ax[1,0].set_xticklabels(['300', '600', '900', '1200', '1500'])
        ax[0,1].imshow(img[img.shape[0]//2:,:])
        ax[1,1].plot(histogram2)
        ax[1,1].set_xticks([0,300,600,900,1200])
        ax[1,1].set_xticklabels(['300', '600', '900', '1200', '1500'])
        ax[0,0].set_title('img warped')
        ax[1,0].set_title('img collapsed')
        ax[0,1].set_title('img warped bottom half')
        ax[1,1].set_title('img collapsed bottom half')

        fig.savefig(os.path.join(save_path,'__collapsed_day_test%d.jpg'%i))
        plt.close(fig)

def pipline_img(img,M,Minv):
    '''img 为已经二值化的经过滤波的图像'''
    thresholded = utils.thresholding(img)
    thresed_warped = Mtrans.warper(thresholded,M)
    # find line points
    left_fit, right_fit, left_lane_inds, right_lane_inds,leftx_base,rightx_base = find_line(thresed_warped)
    cur,dst = utils.calculate_curv_and_pos(thresed_warped,left_fit, right_fit)
    # df = np.concatenate([left_fit.reshape(1,-1), right_fit.reshape(1,-1)],axis = 0)
    # print(df,left_fit.reshape(1,-1))
    res = utils.draw_area(img,thresed_warped,Minv,left_fit,right_fit)
    return utils.draw_values(res,cur,dst)
    

def pipline_video(M,Minv,video_path = 'BY1421_init.mp4',save_path = 'BY1421_2.mp4'):
    file_form = save_path.split('.')[-1]
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if file_form == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif file_form == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    video_out = cv2.VideoWriter()
    video_out.open(save_path,fourcc,fps,size,True)

    for i in tqdm(range(int(frames_num))):
        ret, frame = video.read()
        if ret:
            frame[960:1060,60:210] = 20
            frame_processed = pipline_img(frame,M,Minv)
            video_out.write(frame_processed)
        else:
            print('read video error!')
            break
    print('Finished!')
    video_out.release()
    video.release()
    cv2.destroyAllWindows()

def cal_distance(M,Minv,video_path,save_path):
    '''利用人工标定的车道线的坐标变换，结合这里的检测结果，进行真实偏离距离的评估
    为节约时间，不保存各个车道线识别结果，直接在线计算，保存结果
    '''
    video = cv2.VideoCapture(video_path)
    frames_num = video.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in tqdm(range(int(frames_num))):
        '''找到原来的手动标定的结果的点坐标的txt文件，如果不存在pass;
        如果存在，则进行计算'''
        ret, frame = video.read()
        if ret:
            frame[960:1060,60:210] = 20
            thresholded = utils.thresholding(frame)
            thresed_warped = Mtrans.warper(thresholded,M)
            # find line points
            left_fit, right_fit, left_lane_inds, right_lane_inds,leftx_base,rightx_base = find_line(thresed_warped)
            df = np.concatenate([left_fit.reshape(1,-1), right_fit.reshape(1,-1)],axis = 0)
            np.savetxt(os.path.join(save_path,'%d.txt' %i),df)
        else:
            print('read video error!')
            break
    print('Finished!')
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    M = np.loadtxt('Output/M.txt')
    Minv = np.loadtxt('Output/Minv.txt')
    # pathx = os.path.join('lane_infos','test') 
    # if os.path.exists(pathx) ==  False:
    #     os.mkdir(pathx)
    # save_laneinfo(M,Minv,'BYR0084.mp4',pathx)
    # print(M)
    '''原图坐标点进行透视变换得到的垂直视角的变化'''
    import cal_mtx as Mtrans
    import utils 

    img = cv2.imread('cal_dist/frame_599.jpg')
    left_points = np.loadtxt('cal_dist/frame599_1.txt',delimiter=',').reshape(2,-1)
    right_points = np.loadtxt('cal_dist/frame599_2.txt',delimiter=',').reshape(2,-1)
    # print(left_points)
    left_points_3d = np.concatenate([left_points,np.ones((1,left_points.shape[1]))],axis=0)
    right_points_3d = np.concatenate([right_points,np.ones((1,right_points.shape[1]))],axis=0)

    vertical_lefts = np.dot(M,left_points_3d)
    vertical_rights = np.dot(M,right_points_3d)
    '''standerize'''
    vertical_lefts = vertical_lefts/vertical_lefts[2,:]
    vertical_rights = vertical_rights/vertical_rights[2,:]

    vertical_left_points = vertical_lefts[:2,:].reshape(-1,2)
    vertical_right_points = vertical_rights[:2,:].reshape(-1,2)
    
    print(vertical_left_points)
    # for i in range(vertical_left_points.shape[0]):
    #     cv2.circle(img_warped,tuple(np.around(vertical_left_points[i,:]).astype(np.int)),5,(0,0,255),-1)

    # for i in range(left_points.reshape(-1,2).shape[0]):
    #     cv2.circle(img,tuple(np.around(left_points.reshape(-1,2)[i,:]).astype(np.int)),5,(0,0,255),-1)
    # for i in range(vertical_right_points.shape[0]):
    #     cv2.circle(img_warped,tuple(np.around(vertical_right_points[i,:]).astype(np.int)),10,(0,0,255))
    

    '''以下证明该计算方法是正确的，只需要，保证原始的手动标定的坐标符合有要求即可'''
    lp = np.loadtxt('cal_dist/left_points.txt')
    print(lp)
    lp_3d = np.concatenate([lp.T,np.ones((1,2))],axis = 0)
    lps = np.dot(M,lp_3d)
    lps = lps/lps[2,:]
    print(lps)

    lpv = lps[:2,:].T
    print(lpv)


    img_warped = Mtrans.warper(img,M)
    cv2.circle(img,tuple(np.around(lp[0,:]).astype(np.int)),5,(0,0,255),-1)
    cv2.circle(img,tuple(np.around(lp[1,:]).astype(np.int)),5,(0,0,255),-1)
    cv2.circle(img_warped,tuple(np.around(lpv[0,:]).astype(np.int)),5,(0,0,255),-1)
    cv2.circle(img_warped,tuple(np.around(lpv[1,:]).astype(np.int)),5,(0,0,255),-1)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("image", 1920, 1080)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("image", 1920, 1080)
    cv2.imshow('image',img_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


