import cv2
import os
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import PIL
from PIL import Image

#init points
coor_l1 = np.zeros((1,2))
coor_r1 = np.zeros((1,2))
coor_l2 = np.zeros((1,2))
coor_r2 = np.zeros((1,2))

def OnMouseAction1(event,x,y,flags,param):
    '''初始化参考坐标变换右侧两个点'''
    global coor_r1,coor_r2
    if event == cv2.EVENT_LBUTTONDOWN:
        '''左键按下'''
        coor_r1 = np.array([x, y]).astype(np.int)
    elif event==cv2.EVENT_LBUTTONUP:
        pass
    elif event==cv2.EVENT_RBUTTONDOWN :
        '''右键点击'''
        coor_r2 = np.array([x, y]).astype(np.int)
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        '''左鍵拖曳'''
        pass
    elif event==cv2.EVENT_MBUTTONDOWN :

        '''中键点击'''
        pass
def OnMouseAction2(event,x,y,flags,param):
    '''初始化参考坐标变换左侧两个点'''
    global coor_l1,coor_l2
    if event == cv2.EVENT_LBUTTONDOWN:
        '''左键按下'''
        coor_l1 = np.array([x, y]).astype(np.int)
    elif event==cv2.EVENT_LBUTTONUP:
        pass
    elif event==cv2.EVENT_RBUTTONDOWN :
        '''右键点击'''
        coor_l2 = np.array([x, y]).astype(np.int)
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        '''左鍵拖曳'''
        pass
    elif event==cv2.EVENT_MBUTTONDOWN :
        
        '''中键点击'''
        pass

def confirm_right_coor(img_path):
    img = cv2.imread(img_path)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("image", 1920, 1080)
    cv2.setMouseCallback("image", OnMouseAction1)

    while(True):
        cv2.imshow("image", img)
        cv2.waitKey(0)
        break
        
    cv2.destroyAllWindows()
    points_r = np.row_stack((coor_r1,coor_r2)).astype(np.int)
    return points_r

def confirm_left_coor(img_path,points_r):
    img = cv2.imread(img_path)
    y_up = points_r[0,1]
    y_down = points_r[1,1]
    cv2.line(img,(points_r[0,0],points_r[0,1]),(0,y_up),(0,0,255),thickness = 1)
    cv2.line(img,(points_r[1,0],points_r[1,1]),(0,y_down),(0,0,255),thickness = 1)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("image", 1920, 1080);
    cv2.setMouseCallback("image", OnMouseAction2)

    while(True):
        cv2.imshow("image", img)
        cv2.waitKey(0)
        break
        
    cv2.destroyAllWindows()
    points_l = np.row_stack((coor_l1,coor_l2)).astype(np.int)
    return points_l

'''Perspective transformations'''
def get_M_Minv(points_r,points_l):
    # dash length 2m, space length 4m
    src = np.float32([[(points_l[0][0],points_l[0,1]), (points_l[1,0],points_l[1,1]), (points_r[0,0],points_r[0,1]), (points_r[1,0],points_r[1,1])]])
    dst = np.float32([[(700,0), (700, 1080), (1200, 0), (1200, 1080)]])  #cm 单位
    M = cv2.getPerspectiveTransform(src, dst)    #将原视角转换为俯视
    #source( src ) and destination ( dst ) points
    Minv = cv2.getPerspectiveTransform(dst,src)  #将俯视转换为原视角
    return M,Minv

def warper(img, M):
    # Apply perspective transform
    warped = cv2.warpPerspective(img, M,(img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
def show_res(img,img_warped,savefig = 'perp_res/example_img1.jpg'):
    fig,ax = plt.subplots(1,2, figsize=(16,5))
    b,g,r=cv2.split(img)
    img = cv2.merge([r,g,b])
    b,g,r=cv2.split(img_warped)
    img_warped = cv2.merge([r,g,b])
    ax[0].imshow(img)
    ax[1].imshow(img_warped)
    ax[0].set_title('init img')
    ax[1].set_title('transformed img')

    fig.savefig(savefig)
    plt.close(fig)

if __name__ == '__main__':

    points_r = confirm_right_coor('perp_res/perp_img2.png')
    points_l = confirm_left_coor('perp_res/perp_img2.png',points_r)
    np.savetxt('perp_res/right_points.txt',points_r)
    np.savetxt('perp_res/left_points.txt',points_l)
    M,Minv = get_M_Minv(points_r, points_l)
    np.savetxt('Output/M.txt',M)
    np.savetxt('Output/Minv.txt',Minv)
    
    # points_r = confirm_right_coor('cal_dist/frame_599.jpg')
    # points_l = confirm_left_coor('cal_dist/frame_599.jpg',points_r)
    # np.savetxt('cal_dist/left_points.txt',points_l)
    # np.savetxt('cal_dist/right_points.txt',points_r)


    
