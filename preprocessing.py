# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:22:53 2019

@author: ylHe
"""

import numpy as np

img_x, img_y = (576, 576)
dx = 48

def train_data(imgs, manuals):
    X_train = np.array(imgs)  
    Y_train = np.array(manuals)
    X_train = X_train.astype('float32')/255.
    Y_train = Y_train.astype('float32')/255.
    X_train = X_train[...,1] # the G channel
    X_train = np.array([[X_train[:,v*dx:(v+1)*dx, vv*dx:(vv+1)*dx] for v in range(img_y//dx)] for vv in range(img_x//dx)]).reshape(-1,dx,dx)[:,np.newaxis,...]
    Y_train = np.array([[Y_train[:,v*dx:(v+1)*dx, vv*dx:(vv+1)*dx] for v in range(img_y//dx)] for vv in range(img_x//dx)]).reshape(-1,dx*dx)[...,np.newaxis]
    temp = 1-Y_train
    Y_train = np.concatenate([Y_train,temp],axis=2)
    
    return X_train, Y_train




def test_data(imgs, manuals):
    
    X_test = imgs.astype('float32')/255.
    Y_test = manuals.astype('float32')/255.
    X_test = np.array([[X_test[v*dx:(v+1)*dx, vv*dx:(vv+1)*dx] for v in range(img_y//dx)] for vv in range(img_x//dx)]).reshape(-1,dx,dx)[:,np.newaxis,...]
    
    return X_test, Y_test