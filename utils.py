import numpy as np
import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cal_mtx as Mtrans
from tqdm import tqdm

'''通用工具'''
def get_images_by_dir(dirname):
    '''fetch imgs from folder'''
    img_names = os.listdir(dirname)
    img_paths = [os.path.join(dirname,img_name) for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

'''相机标定与图像失真系数计算  square size 3 cm'''
def findCBCorners(images,grid=(8,6)):
    '''function take the chess board image and return the object points and image points'''
    object_points=[]
    img_points = []
    for img in images:
        #chessboard 是固定的，不妨将实际坐标系固结于上，因此生成的object_points也是固定的一样的54个三维点坐标
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        object_point = object_point*3
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
            #展示角点
            #cv2.drawChessboardCorners(img, grid, corners, True)
    return object_points,img_points

def cal_undistort(img, objpoints, imgpoints,params_save = 'Output/CameraParams.npz'):
    ''' img,实物坐标,图片坐标

    Use cv2.calibrateCamera() and cv2.undistort()
    function takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    
    return:
    mtx:相机参数矩阵
    dist: dist参数
    '''
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    np.savez(params_save, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    return dst,mtx, dist

def show_cal_results(idx,imgs,objp,imgp,output_path = 'Output',grid = [8,6]):
    '''展示chessboard 的查找结果以及失真校准的结果对比图片'''
    #objx = objp[idx]
    imgx = imgp[idx]
    img = imgs[idx]
    status = tqdm(np.arange(imgx.shape[0]))
    print('DrowCornerPoints status:')
    for i,_ in enumerate(status):
        cv2.circle(img,(imgx[i][0,0],imgx[i][0,1]),15,(0,255,0),thickness = 2)
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)
    cv2.imwrite(os.path.join(output_path,'real_img-id%d.jpg'%idx),img)

    udst,mtx,dst = cal_undistort(img, objp, imgp)
    cv2.imwrite(os.path.join(output_path,'undst_img-id%d.jpg'%idx),udst)
    # print(mtx)
    # print(dst)
    test_imgs = get_images_by_dir('CarND review/TEST_IMG')
    status_test = tqdm(test_imgs)
    print('Test status:')
    for idx,imgx in enumerate(status_test):
        undst_img,_,_ = cal_undistort(imgx, objp,imgp)
        b,g,r = cv2.split(imgx) 
        imgx_rgb = cv2.merge([r,g,b]) 

        b_,g_,r_ = cv2.split(undst_img) 
        undst_img_rgb = cv2.merge([r_,g_,b_]) 
        fig,ax = plt.subplots(1,2, figsize=(16,5))
        ax[0].imshow(undst_img_rgb)
        ax[1].imshow(imgx_rgb)
        ax[0].set_title('after undistortion')
        ax[1].set_title('before undistortion')

        fig.savefig(os.path.join(output_path,'test%d.png' % idx))
        cv2.imwrite(os.path.join(output_path,'undst_test_img_%d.jpg'%idx),undst_img)
        cv2.imwrite(os.path.join(output_path,'init_test_img_%d.jpg'%idx),imgx)

'''阀值过滤，生成二值图像'''

def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    '''颜色变化梯度导数过滤'''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    '''全局颜色变化梯度过滤'''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    '''颜色阀值过滤
    HSL 表示 hue（色相）、saturation（饱和度）、lightness（亮度）
    s不适合本数据集，饱和度适合检出良好光照时的黄色
    l 在90-150区间检出效果较好，需要取ROI后进一步避免天空影响
    h 100-102 整体检出黄线的效果较好，黄线较为完整，可以与一个l进一步看效果
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def luv_select(img, thresh=(0, 255)):
    '''L*表示物体亮度，u*和v*是色度'''
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    '''颜色对立空间
    l 亮度,a,b 表示颜色对立维度L分量密切匹配人亮度感知，因此可以被用來通
    过修改a和b分量的輸出色阶來做精確的顏色平衡，或使用L分量來調整亮度對比。a 和b是两个颜
    色通道。a包括的颜色是从深绿色（低亮度值）到灰色（中亮度值）再到亮粉红色（高亮度值）；b
    是从亮蓝色（低亮度值）到灰色（中亮度值）再到黄色（高亮度值）。因此，这种颜色混合后将产生具有明亮效果的色彩。
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_channel = lab[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def thresholding(img,tmax = 0,tmin = 255):
    '''单个滤波器的参数已调整到较优处
    后又将多个组合起来得到较好效果'''
    x_ths = abs_sobel_threshold(img, orient='x', thresh_min=10,thresh_max=100)
    mag_ths = mag_threshold(img, sobel_kernel=9, mag_thresh=(30, 120))
    dir_ths = dir_threshold(img, sobel_kernel=3, thresh=(np.pi/18, np.pi/3.5))
    hls_thresh = hls_select(img,channel = 'h', thresh=(101, 255))
    #hls_thresh_l = hls_select(img,channel = 'l', thresh=(90, 150))
    lab_thresh = lab_select(img, thresh=(100, 150))
    luv_thresh = luv_select(img, thresh=(225, 255))
    #Thresholding combination
    threshholded = np.zeros_like(x_ths)
    #结合多种方式优缺点，得到比较好的结果
    #最终组合： mag and x or hls and dir or lab 参数设置如上
    threshholded[(lab_thresh == 1)|((dir_ths == 1) & (hls_thresh ==1))|((mag_ths == 1)&(x_ths == 1))] = 1
    return threshholded

def test_thres(tmax,tmin,group = 'x and mag or dir and hls' ):
    '''main '''
    test_ths_path = os.path.join('CarND review/thres_test_imgs','filter')
    day_imgs = get_images_by_dir('CarND review/thres_test_imgs/day')
    #night_imgs = get_images_by_dir('CarND review/thres_test_imgs/night')
    if os.path.exists(test_ths_path)==False:
        os.mkdir(test_ths_path)
    #使用两个颜色变化梯度与或一个颜色阀值过滤
    for i,img in enumerate(tqdm(day_imgs)):
        save_path = os.path.join(test_ths_path,group)
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        thsed = thresholding(img,tmax,tmin)

        b,g,r = cv2.split(img) 
        img_rgb = cv2.merge([r,g,b]) 

        # b_,g_,r_ = cv2.split(thsed) 
        # thsed_rgb = cv2.merge([r_,g_,b_]) 
        fig,ax = plt.subplots(1,2, figsize=(16,5))
        ax[0].imshow(thsed,cmap=plt.get_cmap('gray'))
        ax[1].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),cmap=plt.get_cmap('gray'))
        ax[0].set_title('after thresholded')
        ax[1].set_title('before thresholded')

        fig.savefig(os.path.join(save_path,'day_test%d.png' % i))
        plt.close(fig)
    # drop night situation
    # for i,img in enumerate(tqdm(night_imgs)):
    #     save_path = os.path.join(test_ths_path,group)
    #     if os.path.exists(save_path)==False:
    #         os.mkdir(save_path)
    #     thsed = thresholding(img)

    #     b,g,r = cv2.split(img) 
    #     img_rgb = cv2.merge([r,g,b]) 

    #     # b_,g_,r_ = cv2.split(thsed) 
    #     # thsed_rgb = cv2.merge([r_,g_,b_]) 
    #     fig,ax = plt.subplots(1,2, figsize=(16,5))
    #     ax[0].imshow(thsed,cmap=plt.get_cmap('gray'))
    #     ax[1].imshow(img_rgb,cmap=plt.get_cmap('gray'))
    #     ax[0].set_title('after thresholded')
    #     ax[1].set_title('before thresholded')

    #     fig.savefig(os.path.join(save_path,'night_test%d.png' % i))

def test_Mtransform(M):
    '''main '''
    save_path = os.path.join('perp_transform_img','__day_test_Output')
    day_imgs = get_images_by_dir('thres_test_imgs/day')
    #night_imgs = get_images_by_dir('CarND review/thres_test_imgs/night')
    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    
    for i,img in enumerate(tqdm(day_imgs)):

        thsed = thresholding(img)
        img_warped = Mtrans.warper(img,M)
        thsed_warped = Mtrans.warper(thsed,M)
        b,g,r = cv2.split(img) 
        img = cv2.merge([r,g,b]) 
        b,g,r=cv2.split(img_warped)
        img_warped = cv2.merge([r,g,b])
        #cv2.imwrite(os.path.join(save_path,'thsed_warped%d.jpg'%i),thsed_warped)
        
        fig,ax = plt.subplots(2,2, figsize=(16,10))
        ax[0,0].imshow(img)
        ax[0,1].imshow(thsed)
        ax[1,0].imshow(img_warped)
        ax[1,1].imshow(thsed_warped)
        ax[0,0].set_title('before thresholded')
        ax[0,1].set_title('after thresholded')
        ax[1,0].set_title('before transformed')
        ax[1,1].set_title('after transformed')

        fig.savefig(os.path.join(save_path,'day_test%d.png' % i))
        plt.close(fig)

        fig,ax = plt.subplots(figsize=(16,9))
        ax.imshow(thsed_warped)
        #print(thsed_warped.shape)
        cv2.imwrite(os.path.join(save_path,'warped_day_test%d.jpg' % i),thsed_warped*255)
        np.savetxt(os.path.join(save_path,'warped_day_test%d.txt' % i),thsed_warped)
        ax.set_title('img_warped')
        fig.savefig(os.path.join(save_path,'warped_day_test%d.png' % i))
        plt.close(fig)

def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
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


'''save lane data'''
def draw_values(img,curvature,distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of curvature: %sm"%(round(curvature))
    
    # if distance_from_center>0:
    #     pos_flag = 'right'
    # else:
    #     pos_flag= 'left'
        
    cv2.putText(img,radius_text,(100,100), font, 2,(0,0,255),2)
    # center_text = "Vehicle is %.3fm %s of the center"%(abs(distance_from_center),pos_flag)
    # cv2.putText(img,center_text,(100,150), font, 2,(0,0,255),2)
    return img

def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
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

if __name__ == '__main__':

    '''测试阀值参数调节'''
    # tmax = 255
    # for tmin in np.arange(225,226,20):
        
    #     ths_main(tmax,tmin,group = 'luv ch-l min {0} max {1})'.format(tmin,tmax))



    '''测试去失真'''
    # imgs = get_images_by_dir('cal_imgs')
    # objp,imgp = findCBCorners(imgs)

    # show_cal_results(12,imgs,objp,imgp)


    '''测试obj points生成'''
    # grid = [8,6]
    # object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
    # object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
    # print(object_point)

    '''测试chessboard finding及其结果'''
    # imgs = get_images_by_dir('cal_imgs')
    # im = imgs[0]

    # print(im.shape,im.shape[1::-1])
    # print(len(imgs))
    # print(type(imgs[0]))

    # objp,imgp = calibrate(imgs)
    # print('objp和imgp长度：',len(objp),len(imgp))

    # print(im.shape,type(imgp[0]),imgp[0][0].shape,imgp[0].shape)
    # #BGR颜色  最后填充参数，默认不填充，值为-1填充
    # '''circle(img, center, radius, color[, thickness[, lineType[, shift]]])'''
    # cv2.circle(im,(imgp[0][0][0,0],imgp[0][0][0,1]),15,(0,255,0),thickness = 2)
    # cv2.imshow('Test calibrate', im)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    '''测试空间变换'''
    M = np.loadtxt('Output/M.txt')
    test_Mtransform(M)