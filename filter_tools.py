import numpy as np
import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cal_mtx as Mtrans
from tqdm import tqdm

def get_images_by_dir(dirname):
    '''
    Fetch imgs from folder
    Attributes:
        Input:
            - dirname: directory of calibration files
        Output:
            - imgs: list of images(np.array)
    '''
    img_names = os.listdir(dirname)
    img_paths = [os.path.join(dirname,img_name) for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    '''
    Sobel Filter based abs(grad) filtering 
    Attributes:
        Input:
            - img: RGB image 
            - orient: Partial derivative direction 
            - thresh_min
            - thresh_max
        Output:
            - binary_output: binary output
    '''
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
    '''
    Sobel Filter based grad magnitude filtering
    Attributes:
        Input:
            - img: RGB image 
            - sobel_kernel: kernel size of sobel filter
            - mag_thresh: type:tuple, default = (0,255)
        Output:
            - binary_output: binary output
    '''
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
    '''
    Sobel Filter based Partial derivative direction filtering
    Attributes:
        Input:
            - img: RGB image 
            - sobel_kernel: kernel size of sobel filter
            - mag_thresh: type:tuple, default = (0, np.pi/2)
        Output:
            - binary_output: binary output
    '''
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
    '''
    Color space filtering
    Description:
        HSL: hue;saturation;lightness
        s is not suitable for this data set, and the saturation is suitable for detecting yellow under good light
        l The detection effect is better in the 90-150 interval, and the ROI needs to be taken to further avoid \
            the influence of the sky
        h 100-102 The overall detection effect of the yellow line is better, and the yellow line is relatively \
            complete. You can further see the effect with an l
    Attributes:
        Input:
            - img: RGB image 
            - channel: channel to be filtered
            - thresh: type:tuple, default = (0,255)
        Output:
            - binary_output: binary output
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
    '''
    Color space filtering
    Description:
        Luv: L represents the brightness of the object, u and v are the chromaticity
    Attributes:
        Input:
            - img: RGB image 
            - thresh: default=(0,255),type = tuple
        Output:
            - binary_output: binary output
    '''
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    '''
    Color space filtering
    Description:
        Color Opposition Space
            l means brightness, a, b means that the color opposite dimension L component closely matches human \
            brightness perception, so it can be used to make accurate color balance by modifying the output color \
            scale of a and b components, or use L component to adjust brightness contrast. a and b are two color \
            channels. a includes colors ranging from dark green (low brightness value) to gray (medium brightness \
            value) to bright pink (high brightness value); b is from bright blue (low brightness value) to gray \
            (medium brightness value) and then To yellow (high brightness value). Therefore, this kind of color \
            mixing will produce bright colors.
    Attributes:
        Input:
            - img: RGB image 
            - thresh: default=(0,255),type = tuple
        Output:
            - binary_output: binary output
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_channel = lab[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def thresholding(img,tmax = 0,tmin = 255):
    '''
    Filter function combined with several filter operations
    Description:
        The parameters of a single filter have been adjusted to better points, and then multiple combinations \
        can be combined to get better results
    Attributes:
        Input:
            img: RGB image;
            tmax: minimum threshold
            tmin: maximum threshold
    '''
    x_ths = abs_sobel_threshold(img, orient='x', thresh_min=10,thresh_max=100)
    mag_ths = mag_threshold(img, sobel_kernel=9, mag_thresh=(30, 120))
    dir_ths = dir_threshold(img, sobel_kernel=3, thresh=(np.pi/18, np.pi/3.5))
    hls_thresh = hls_select(img,channel = 'h', thresh=(101, 255))
    #hls_thresh_l = hls_select(img,channel = 'l', thresh=(90, 150))
    lab_thresh = lab_select(img, thresh=(100, 150))
    luv_thresh = luv_select(img, thresh=(225, 255))
    #Thresholding combination
    threshholded = np.zeros_like(x_ths)

    #final combination： mag and x or hls and dir or lab, params as above
    threshholded[(lab_thresh == 1)|((dir_ths == 1) & (hls_thresh ==1))|((mag_ths == 1)&(x_ths == 1))] = 1
    return threshholded

def eval_threshold_params(tmax,tmin,test_imgs_dir = None,group = 'x and mag or dir and hls' ):
    '''
    Filter function params fine tune
    Attributes:
        Input:
            tmax: minimum threshold of specified parameter
            tmin: maximum threshold of specified parameter
            test_imgs_dir: images for parameters fine tune,default=None
            group: combination of filters, 'x and mag or dir and hls'
    '''
    test_ths_path = './filter_params_fine_tune'
    if os.path.exists(test_ths_path) == False:
        os.mkdir(test_ths_path)
    if os.path.exists(test_imgs_dir) == False:
        raise FileNotFoundError
    imgs = get_images_by_dir(test_imgs_dir)
    if os.path.exists(test_ths_path)==False:
        os.mkdir(test_ths_path)
    for i,img in enumerate(tqdm(imgs)):
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


if __name__ == '__main__':

    pass