# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:41:41 2019

@author: ylHe
"""
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from preprocessing import test_data
import cv2
import numpy as np

img_x, img_y = (576, 576)
dx = 48

imgs_test = cv2.imread('DRIVE/test/images/01_test.tif')[...,1] #the G channel
imgs_test = cv2.resize(imgs_test,(img_x, img_y))
manuals_test = np.asarray(Image.open('DRIVE/test/1st_manual/01_manual1.gif'))

X_train, Y_train = train_data(imgs_train, manuals_train)
X_test, Y_test = test_data(imgs_test, manuals_test)


model = load_model('best_weights.h5')
Y_pred = model.predict(X_test)
Y_pred = Y_pred[...,0].reshape(img_x//dx,img_y//dx,dx,dx)
Y_pred = [Y_pred[:,v,...] for v in range(img_x//dx)]
Y_pred = np.concatenate(np.concatenate(Y_pred,axis=1),axis=1)
Y_pred = cv2.resize(Y_pred,(Y_test.shape[1], Y_test.shape[0]))
plt.figure(figsize=(6,6))
plt.imshow(Y_pred)
plt.figure(figsize=(6,6))
plt.imshow(Y_test)