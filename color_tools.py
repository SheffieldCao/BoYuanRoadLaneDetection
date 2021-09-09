import cv2
import numpy as np

def select_yellow(image):
    '''
    Generate mask of specified color(yellow)
    Attributes:
        Input:
            - image: RGB image
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def select_white(image):
    '''
    Generate mask of specified color(yellow)
    Attributes:
        Input:
            - image: RGB image
    '''
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    
    return mask

