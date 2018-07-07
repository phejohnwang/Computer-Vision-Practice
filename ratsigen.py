# -*- coding: utf-8 -*-
"""
@author: pheno

Data Generator for RatSI dataset
    RatSI: Rat Social Interaction Dataset by Noldus
    https://www.noldus.com/projects/phenorat/datasets/ratsi

Input
    image: folder with image frames generated from original videos
        folder_path/
            Sample000XXX.jpg
            Sampel000XXX.jpg
            ......
    label: original csv file
     
"""

from tensorflow.python.keras.utils import to_categorical
import os
from glob import glob
import numpy as np
import pandas as pd
import cv2
# Use Opencv 3 for image processing

class RatSIDataGenerator(object):

    def __init__(self,
                 batch_size=32,
                 flip_prob=0.5,
                 rotation = False,
                 rotation_range=180,
                 base = 'InceptionV3'):
        self.batch_size = batch_size
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.rotation_range = rotation_range
        self.depth = 3
        if base not in {'InceptionV3', 'VGG'}:
            raise ValueError('Unknown base '+str(base))
        self.base = base
        if self.base == 'InceptionV3':
            self.height = 299
            self.width = 299
        elif self.base == 'VGG':
            self.height = 224
            self.width = 224

        self.Action_names = ['Allogrooming','Approaching','Following','Moving away','Nape attacking',
                        'Pinning','Social Nose Contact','Solitary','Other','Uncertain']
        self.dataset_root = 'D:/RatSI/RatSI'
        self.y_filename = sorted(glob(os.path.join(self.dataset_root, 'annotations', 'Observation*.csv')))      
        self.total_frames = [21514, 21334, 22840, 22742, 22858, 22751, 22923, 22771, 22715]
        self.train_list = [1,2,3,4,8,9]
        self.test_list = [5,6,7]
        
        # Index starts with 0
        self.x_train_index = {} #actual frame number - 1
        self.y_train = {}
        
        self.x_test_index = {}
        self.y_test = {}
        
        self.get_data_train()
        self.modify_data_train()
        
        self.get_data_test()
        self.modify_data_test()
        
        #self.combine_data()        # no need to combine data during initialization
        self.y_train_cate = {}
        self.y_test_cate = {}
        self.get_categorical()
        
        self.get_steps_per_epoch()

    def label2number(self, truth_label):
        # Covnert str label to number truth
        """
        0 - Allogrooming
        1 - Approaching
        2 - Following
        3 - Moving away
        4 - Nape attacking
        5 - Pinning
        6 - Social Nose Contact
        7 - Solitary
        8 - Other
        9 - Uncertain
        """
        truth_number = np.zeros(len(truth_label),dtype=np.int32)
        for i in range(len(self.Action_names)):
            map_ = truth_label == self.Action_names[i]
            map_int = np.array(map_,dtype=np.int32) # true = 1, false = 0
            truth_number = truth_number + map_int * i
        return truth_number

    # get truth from one observation
    def get_annotations(self, filename, total_size):
        behavior_data = pd.read_csv(filename, sep=';', index_col=0)
        tmp = behavior_data.values
        behavior_data_label = tmp.reshape(len(tmp))
        annotations = self.label2number(behavior_data_label)
        annotations = annotations[:total_size]
        return annotations

    def get_data_train(self):
        # read videos indicated by train_list
        for i in self.train_list:
            self.x_train_index[i] = np.arange(self.total_frames[i-1])
            self.y_train[i] = self.get_annotations(self.y_filename[i-1], self.x_train_index[i].shape[0])

    def modify_data_train(self):
        for i in self.train_list:
            # remove '9 - Uncertain'
            mask = self.y_train[i] != 9
            self.y_train[i] = self.y_train[i][mask]
            self.x_train_index[i] = self.x_train_index[i][mask]
            # randomly sample 30% of '7 - Solitary' / remove 70% of '7 - Solitary'
            modify_index = np.argwhere(self.y_train[i]==7)
            np.random.shuffle(modify_index)
            select_index = modify_index[:int(0.7*len(modify_index))]
            self.y_train[i] = np.delete(self.y_train[i], select_index)
            self.x_train_index[i] = np.delete(self.x_train_index[i], select_index)

    def get_data_test(self):
        # read videos indicated by test_list
        for i in self.test_list:
            self.x_test_index[i] = np.arange(self.total_frames[i-1])
            self.y_test[i] = self.get_annotations(self.y_filename[i-1], self.x_test_index[i].shape[0])        
    
    def modify_data_test(self):
        # for test data, no need to resample '7 - Solitary'
        for i in self.test_list:
            # remove '9 - Uncertain'
            mask = self.y_test[i] != 9
            self.y_test[i] = self.y_test[i][mask]
            self.x_test_index[i] = self.x_test_index[i][mask]

    def get_categorical(self):
        for i in self.train_list:
            self.y_train_cate[i] = to_categorical(self.y_train[i], num_classes=9)
        for i in self.test_list:
            self.y_test_cate[i] = to_categorical(self.y_test[i], num_classes=9)
     
    def get_steps_per_epoch(self):
        self.steps_per_epoch_train = 0
        for i in self.train_list:
            self.steps_per_epoch_train += len(self.y_train[i]) // self.batch_size            
        self.steps_per_epoch_test = 0
        for i in self.test_list:
            self.steps_per_epoch_test += len(self.y_test[i]) // self.batch_size       
        
    def rotate(self, number):
        # Rotate image - counterclockwise
        pass
    
    def flip(self, number):
        # To be implemented   
        pass

    def do_shuffle_train(self):
        np.random.shuffle(self.train_list)
        # shuffle x & y in the same way
        for i in self.train_list:
            tmp = np.arange(len(self.y_train[i]))
            np.random.shuffle(tmp)
            self.x_train_index[i] = self.x_train_index[i][tmp]
            self.y_train[i] = self.y_train[i][tmp]
            self.y_train_cate[i] = self.y_train_cate[i][tmp]

    def generate(self, shuffle=True):
        while True:
            # shuffle frames within video and also video order 
            if shuffle:
                self.do_shuffle_train()
            # generate training batch
            # skip the last batch if it is less than batch_size
            for j in self.train_list:
                cuts = [(b, min(b + self.batch_size, len(self.y_train[j]))) for b in range(0, len(self.y_train[j]), self.batch_size)]
                for start, end in cuts:
                    if (end - start)!=self.batch_size:
                        continue
                    self.images = np.zeros((self.batch_size,self.height,self.width,self.depth), dtype=np.float32)
                    self.targets = np.zeros((self.batch_size, 9), dtype=np.float32)
                    for i in range(start,end):
                        # Get image and annoation
                        img_path = self.dataset_root + '/videos/' + '%d/Sample%06d.png' % (j, self.x_train_index[j][i]+1)
                        img = cv2.imread(img_path)  # read as color image
                        img = cv2.resize(img, (self.width,self.height))
                        x = np.array(img, dtype=np.float32)
                        self.images[i-start] = x
                        self.targets[i-start] = self.y_train_cate[j][i]
                        # Rotate image with random angle
                        #if self.rotation:
                            #self.rotate(i-start)
                        #self.flip(i-start)
                        # preprcess input as required
                    if self.base == 'Inception':
                        self.images /= 127.5
                        self.images -= 1.
                    elif self.base == 'VGG':
                        self.images[..., 0] -= 103.939
                        self.images[..., 1] -= 116.779
                        self.images[..., 2] -= 123.68
                    yield (self.images, self.targets)                        

    def generate_test(self):
        while True:
            for j in self.test_list:
                cuts = [(b, min(b + self.batch_size, len(self.y_test[j]))) for b in range(0, len(self.y_test[j]), self.batch_size)]
                for start, end in cuts:
                    if (end - start)!=self.batch_size:
                        continue
                    self.test_images = np.zeros((self.batch_size,self.height,self.width,self.depth), dtype=np.float32)
                    self.test_targets = np.zeros((self.batch_size, 9), dtype=np.float32)
                    for i in range(start,end):
                        # Get image and annoation
                        img_path = self.dataset_root + '/videos/' + '%d/Sample%06d.png' % (j, self.x_test_index[j][i]+1)
                        img = cv2.imread(img_path)  # read as color image
                        img = cv2.resize(img, (self.width,self.height))
                        x = np.array(img, dtype=np.float32)
                        self.test_images[i-start] = x
                        self.test_targets[i-start] = self.y_test_cate[j][i]
                        # Rotate image with random angle
                        #if self.rotation:
                            #self.rotate(i-start)
                        #self.flip(i-start)
                        # preprcess input as required
                    if self.base == 'Inception':
                        self.test_images /= 127.5
                        self.test_images -= 1.
                    elif self.base == 'VGG':
                        self.test_images[..., 0] -= 103.939
                        self.test_images[..., 1] -= 116.779
                        self.test_images[..., 2] -= 123.68
                    yield (self.test_images, self.test_targets) 

    def generate_test_x_only(self):
        while True:
            for j in self.test_list:
                cuts = [(b, min(b + self.batch_size, len(self.y_test[j]))) for b in range(0, len(self.y_test[j]), self.batch_size)]
                for start, end in cuts:
                    if (end - start)!=self.batch_size:
                        continue
                    self.test_images = np.zeros((self.batch_size,self.height,self.width,self.depth), dtype=np.float32)
                    #self.test_targets = np.zeros((self.batch_size, 9), dtype=np.float32)
                    for i in range(start,end):
                        # Get image and annoation
                        img_path = self.dataset_root + '/videos/' + '%d/Sample%06d.png' % (j, self.x_test_index[j][i]+1)
                        img = cv2.imread(img_path)  # read as color image
                        img = cv2.resize(img, (self.width,self.height))
                        x = np.array(img, dtype=np.float32)
                        self.test_images[i-start] = x
                        #self.test_targets[i-start] = self.y_test_cate[j][i]
                        # Rotate image with random angle
                        #if self.rotation:
                            #self.rotate(i-start)
                        #self.flip(i-start)
                        # preprcess input as required
                    if self.base == 'Inception':
                        self.test_images /= 127.5
                        self.test_images -= 1.
                    elif self.base == 'VGG':
                        self.test_images[..., 0] -= 103.939
                        self.test_images[..., 1] -= 116.779
                        self.test_images[..., 2] -= 123.68
                    yield (self.test_images)                     