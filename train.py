# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:40:38 2019

@author: ylHe
"""

from unet import unet_model
from preprocessing import train_data, test_data
import numpy as np
import cv2
from PIL import Image
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

img_x, img_y = (576, 576)
dx = 48

filelst = os.listdir('DRIVE/training/images/')
filelst = ['DRIVE/training/images/'+v for v in filelst]
imgs_train = [cv2.imread(file) for file in filelst]
filelst = os.listdir('DRIVE/training/1st_manual/')
filelst = ['DRIVE/training/1st_manual/'+v for v in filelst]
manuals_train = [np.asarray(Image.open(file)) for file in filelst]
imgs_train = [cv2.resize(v,(img_x, img_y)) for v in imgs_train]
manuals_train = [cv2.resize(v,(img_x, img_y)) for v in manuals_train]

imgs_test = cv2.imread('DRIVE/test/images/01_test.tif')[...,1] #the G channel
imgs_test = cv2.resize(imgs_test,(img_x, img_y))
manuals_test = np.asarray(Image.open('DRIVE/test/1st_manual/01_manual1.gif'))


X_train, Y_train = train_data(imgs_train, manuals_train)
X_test, Y_test = test_data(imgs_test, manuals_test)

model = unet_model(X_train.shape[1],X_train.shape[2],X_train.shape[3])
model.summary()

checkpointer = ModelCheckpoint(filepath='best_weights.h5', verbose=1, monitor='val_acc', 
                              mode='auto', save_best_only=True)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=64, epochs=20, verbose=2,shuffle=True, validation_split=0.2,
                 callbacks=[checkpointer])


