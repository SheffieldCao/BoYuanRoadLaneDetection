import numpy as np
import cv2

def findCBCorners(images,grid=(8,6)):
    '''
    Find the object points and image points on the chess board;
    Attributes:
        Input:
            - images: list of img(np.array)
            - grid: chessboard grid size, type: tuple, default=(8,6)
        Outputs:
            - object_points: world coordinates of chessboard corners
            - img_points: pixel coordinates of chessboard corners
    '''
    object_points=[]
    img_points = []
    for img in images:
        #assume the world coordinates of chessboard are fixed and specified
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        object_point = object_point*3
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
            # whether to show the corners
            # cv2.drawChessboardCorners(img, grid, corners, True)
    return object_points,img_points

def cal_undistort(img, objpoints, imgpoints,params_save = 'calibrate_res/CameraParams.npz'):
    '''
    Calculate distortion factor.
    Description:
        Use cv2.calibrateCamera() and cv2.undistort()
        function takes an image, object points, and image points
        performs the camera calibration, image distortion correction and 
        returns the undistorted image
    Attributes:
        Input:
            - img: RGB input image
            - objpoints: world coordinates of chessboard corners
            - imgpoints: pixel coordinates of chessboard corners
        Output:
            - dst: undistorted image
            - mtx: camera matrix
            - distoration: distoration factor
    '''
    ret, mtx, distoration, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    if not ret:
        raise ValueError
    dst = cv2.undistort(img, mtx, distoration, None, mtx)
    np.savez(params_save, mtx=mtx, dist=distoration, rvecs=rvecs, tvecs=tvecs)
    return dst, mtx, distoration
        

def get_M_Minv(points_r,points_l):
    '''
    Calculate perspective transform matrix.
    Description:
        Considering that the lane line design are standard that 'dash' length is 2m and 'space' length is 4m;
        So choosing different numbers of dashes, we got different transform matrix with respect to different forward distance 
    Attributes:
        Inputs:
            - points_r: chosed right lane line points based on a rectangle on the ground in the world coordinate system.
            - points_l: chosed left lane line points based on a rectangle on the ground in the world coordinate system.
        Outputs:
            - M: perspective transform matrix(from vehicle view to bird's eye view)
            - Minv: inverse perspective transform matrix of M
    '''
    # dash length 2m, space length 4m
    src = np.float32([[(points_l[0][0],points_l[0,1]), (points_l[1,0],points_l[1,1]), (points_r[0,0],points_r[0,1]), (points_r[1,0],points_r[1,1])]])
    # Specify the target scale under the bird's-eye view
    dst = np.float32([[(700,0), (700, 1080), (1200, 0), (1200, 1080)]]) 
    M = cv2.getPerspectiveTransform(src, dst)    #from vehicle view to bird's eye view
    #source( src ) and destination ( dst ) points
    Minv = cv2.getPerspectiveTransform(dst,src)  #from bird's eye view to vehicle view
    return M,Minv

def warper(img, M):
    '''
    Perform image warping.
    Attributes:
        Inputs:
            - img: RGB image in vehicle view
            - M: perspective transform matrix(from vehicle view to bird's eye view)
        Outputs:
            - warped: img warped by M
    '''
    # Apply perspective transform
    # keep same size as input image
    warped = cv2.warpPerspective(img, M,(img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  
    return warped