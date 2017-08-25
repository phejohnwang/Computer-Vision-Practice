# -*- coding: utf-8 -*-
"""
Version: 2017-8-25

@author: pheno

Convert video into image frames (Sample00XXX.png)
    Usage: python Video2Image.py -g Trial_1_2017_07_12_13_55_35.avi ./Depth
    
    Currently set FPSCounter = 1 for full frames conversion

"""

import argparse
import cv2

def main(video_file_name, output_folder_path, grayscale):
    FPSCounter = 1
    SampleCounter = 1
    cap = cv2.VideoCapture(video_file_name)
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if FPSCounter == 1:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            img_path = output_folder_path + '/Sample%05d.png' % SampleCounter
            cv2.imwrite(img_path, frame)
            if SampleCounter%300==0:
                print(int(SampleCounter/300))
            SampleCounter = SampleCounter + 1
            FPSCounter = 1
        else:
            FPSCounter = FPSCounter + 1
    
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert video into image frames (Sample00XXX.png)")
    parser.add_argument("-g", "--grayscale", action="store_true", help="save as 8 bit grayscale")
    parser.add_argument("input", help="input video file name")
    parser.add_argument("output", help="output folder path")
    args = parser.parse_args()
    if args.grayscale:
        grayscale = True
    else:
        grayscale = False

    main(args.input, args.output, grayscale)
