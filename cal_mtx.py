import cv2
import os
import numpy as np
from cam_tools import *

#init points
coor_l1 = np.zeros((1,2))
coor_r1 = np.zeros((1,2))
coor_l2 = np.zeros((1,2))
coor_r2 = np.zeros((1,2))

def OnMouseAction1(event,x,y,flags,param):
    '''
    Chose right points by using OnMouseAction
    '''
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
    '''
    Chose left points by using OnMouseAction
    '''
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
    '''
    Chose left points by using OnMouseAction
    for more details about left-right points, go to Description of func:cam_tools.get_M_Minv
    '''
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
    '''
    Chose right points by using OnMouseAction
    for more details about left-right points, go to Description of func:cam_tools.get_M_Minv
    '''
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

if __name__ == '__main__':

    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Camera calibration."
    )

    arg_parser.add_argument('--path', type=str, default='straight_lane_samples/perp_img2.png', help='path to perspective transform calibration image')
    arg_parser.add_argument('--save_M', type=str, default='calibrate_res', help='Path to save calibration results, including M,Minv') 
    
    args = arg_parser.parse_args()

    points_r = confirm_right_coor(args.path)
    points_l = confirm_left_coor(args.path,points_r)
    np.savetxt(os.path.join(args.save_M,'right_points.txt'),points_r)
    np.savetxt(os.path.join(args.save_M,'left_points.txt'),points_l)
    M,Minv = get_M_Minv(points_r, points_l)
    np.savetxt(os.path.join(args.save_M,'M.txt'),M)
    np.savetxt(os.path.join(args.save_M,'Minv.txt'),Minv)
    
