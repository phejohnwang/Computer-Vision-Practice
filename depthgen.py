# -*- coding: utf-8 -*-
"""

@author: pheno

Generator for data augmentation on depth images

Input
    image: folder with cropped 160*160 grayscale images
        x_train_folder_path/
            Sample00001.png
            Sampel00002.png
            ......
    label: txt file with cropped coordinates (before normalization)
    
Inspired by @moannuo
    https://github.com/moannuo/goodtoknow/blob/master/ML/Kaggle/KeypointsFacialKeras/generator.py
    
    for rotation, one thing to notice is that the direction of y axis is (top to bottom)   
"""

#from tensorflow.contrib.keras.python.keras.preprocessing import image
import numpy as np
import cv2
# Use Opencv 3 for image processing

class DepthDataGenerator(object):

    def __init__(self,
                 x_train_folder_path,
                 y_train_file_path,
                 total_size,
                 batch_size=32,
                 flip_prob=0.5,
                 rotation = True,
                 rotation_range=180):
        self.x_train_folder_path = x_train_folder_path
        self.y_train_file_path = y_train_file_path
        self.total_size = total_size
        self.batch_size = batch_size
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.rotation_range = rotation_range
        # Index for image reading
        self.index = np.arange(total_size)
        self.width = 128
        self.height = 128
        self.depth = 1
        # Read truth annotations
        self.y_train = np.loadtxt(y_train_file_path, dtype=np.float32)
        self.y_train = self.y_train[:,1:]
        # Coordinates normalization w.r.t. the bounding box b = (bc,bw,bh)
        # b select as the whole image
        # bc = (64,64), bw = 128, bh = 128
        # y has already been centered before reading
        self.y_train = self.y_train / self.width
        
    def rotate(self, number):
        # Rotate image - counterclockwise
        angle = np.random.randint(0, self.rotation_range)
        M = cv2.getRotationMatrix2D((self.width/2,self.height/2),angle,1)
        dst = cv2.warpAffine(self.images[number,:,:,0],M,(self.width,self.height),borderValue = -1)
        self.images[number] = dst.reshape(dst.shape[0], dst.shape[1],1)
        # Rotate coordinates - counterclockwise
        # R matrix angle is clockwise as y axis is pointing upside down
        angle_R = np.radians(360-angle)
        R = np.array([[np.cos(angle_R), np.sin(angle_R)], [-np.sin(angle_R), np.cos(angle_R)]])
        y_temp = self.targets[number]
        y_temp = y_temp.reshape(-1,2)
        y_temp = np.dot(y_temp,R)
        y_temp = y_temp.reshape(-1)
        self.targets[number] = y_temp
    
    def flip(self, number):
        # To be implemented   
        pass
        
    def generate(self, shuffle=True):
        while True:
            if shuffle:
                np.random.shuffle(self.index)
            cuts = [(b, min(b + self.batch_size, self.total_size)) for b in range(0, self.total_size, self.batch_size)]
            for start, end in cuts:
                #namelist = []
                self.images = np.zeros((self.batch_size,self.height,self.width,self.depth), dtype=np.float32)
                self.targets = np.zeros((self.batch_size, 6), dtype=np.float32)
                for i in range(start,end):
                    # Get image and annoation
                    #img_path = './Data_depth/V13_17_43/Sample%05d.jpg' % (self.index[i]+1)
                    img_path = self.x_train_folder_path + '/Sample%05d.png' % (self.index[i]+1)
                    #namelist.append(img_path)
                    img = cv2.imread(img_path,0)           
                    x = np.array(img, dtype=np.float32)
                    # Map 8 bit grayscale to [-1,1]
                    x = x / 255 - 0.5
                    x = x * 2
                    x = x.reshape(x.shape[0],x.shape[1],1)
                    self.images[i-start] = x                 
                    self.targets[i-start] = self.y_train[self.index[i]]
                    # Rotate image with random angle
                    if self.rotation:
                        self.rotate(i-start)
                    #self.flip(i-start)
                #self.inputs = namelist
                yield (self.images, self.targets)

