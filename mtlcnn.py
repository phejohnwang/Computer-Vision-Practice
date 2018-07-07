# -*- coding: utf-8 -*-
"""

@author: pheno

MTL CNN Architecture - VGG16

    Use tf.keras
    
    Use GlobalAveragePooling2D

"""

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import concatenate

#----------------------Build CNN Model---------------------------------------
def MTLCNN(include_depth = False):
    # main conv module - color
    # --- input: preprocessed rodent color crop (224x224x3)
    # --- output: shared color features (None,512)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')
    
    # side conv moudle - depth
    # --- input: preprocessed rodent depth crop (64x64)
    # --- output: shared depth features (None,64)
    # [Block] = [CONV + CONV + POOL]
    # [Block] * 2
    if include_depth == True:
        depth_input = Input(shape=(64,64,1),name = 'depth_input')
        xd = Conv2D(16, (3, 3), activation='relu', padding='same', name='depthb1_conv1')(depth_input)
        xd = Conv2D(16, (3, 3), activation='relu', padding='same', name='depthb1_conv2')(xd)
        xd = MaxPooling2D((2, 2), strides=(2, 2), name='depthb1_pool')(xd)
        # 16x32x32
        xd = Conv2D(32, (3, 3), activation='relu', padding='same', name='depthb2_conv1')(xd)
        xd = Conv2D(32, (3, 3), activation='relu', padding='same', name='depthb2_conv2')(xd)
        xd = MaxPooling2D((2, 2), strides=(2, 2), name='depthb2_pool')(xd)
        # 32*16*16
        xd = Conv2D(64, (3, 3), activation='relu', padding='same', name='depthb3_conv1')(xd)
        xd = Conv2D(64, (3, 3), activation='relu', padding='same', name='depthb3_conv2')(xd)
        xd = MaxPooling2D((2, 2), strides=(2, 2), name='depthb3_pool')(xd)
        # 64*8*8
        xd = GlobalAveragePooling2D()(xd)
        # (none,64)
        shared_feature = concatenate([base_model.output, xd])
        # use lower case concatenate
    else:
        shared_feature = base_model.output

    # pose estimation branch
    x = Dense(512, activation='relu', name='pose_fc1')(shared_feature)
    x = Dense(256, activation='relu', name='pose_fc2')(x)
    pose = Dense(14, name='pose_predict')(x)

    # behavior recognition branch - body
    y = Dense(256, activation='relu', name='body_fc1')(shared_feature)
    body = Dense(4, activation='softmax', name='body_predict')(y)

    # behavior recognition branch - head
    z = Dense(256, activation='relu', name='head_fc1')(shared_feature)
    head = Dense(4, activation='softmax', name='head_predict')(z)

    # combine all branches
    if include_depth == True:
        model = Model(inputs=[base_model.input, depth_input], outputs=[pose, body, head])
    else:
        model = Model(inputs=base_model.input, outputs=[pose, body, head])

    return model


"""
Usage

a = MTLCNN()
a.compile(optimizer='rmsprop',
              loss={'pose_predict': 'mean_squared_error', 
                    'body_predict': 'categorical_crossentropy',
                    'head_predict': 'categorical_crossentropy'},
              loss_weights={'pose_predict': 1., 'body_predict': 0.5, 'head_predict':0.5})
"""






