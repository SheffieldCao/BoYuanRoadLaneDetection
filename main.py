from lane_detect import *
from filter_tools import get_images_by_dir
import cam_tools
import os
import argparse

def main():
    arg_parser = argparse.ArgumentParser(
        description="Lane Detection."
    )

    arg_parser.add_argument('--path', type=str, default='data/example.mp4', help='path to video or images to detect lane')
    arg_parser.add_argument('--save_path', type=str, default='examples/demo_example.mp4', help='path to save video or images with the detected lane')
    arg_parser.add_argument('--task', type=str, default='video', help='Task to perform: video, img, cal_camera') 
    arg_parser.add_argument('--M', type=str, default='calibrate_res/M.txt', help='path of M') 
    arg_parser.add_argument('--Minv', type=str, default='calibrate_res/Minv.txt', help='path of Minv') 
    args = arg_parser.parse_args()
    M = np.loadtxt(args.M)
    Minv = np.loadtxt(args.Minv)
    if os.path.exists(args.path) == False:
        raise FileNotFoundError
    if args.path.split('.')[-1] == 'mp4' and args.task == 'video':
        detect_video(M,Minv,args.path,args.save_path)
    elif args.path.split('.')[-1] != 'mp4' and args.task == 'img' :
        img = cv2.imread(args.path)
        IMG = detect_img(img,M,Minv)
        cv2.imwrite(args.save_path,IMG)
    else:
        raise ValueError('Mismatched tasks')

if __name__ == '__main__':
    main()