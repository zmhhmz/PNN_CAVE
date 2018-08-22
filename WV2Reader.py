# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 22:20:12 2018

@author: XieQi
"""

import numpy as np
import scipy.io as sio  
#import MyLib as ML
import random 
#import cv2
def all_train_data_in():
    data = sio.loadmat('WV2Data/trainXYZ')
    allDataX = data['CX']
    allDataY = data['CY']
    return allDataX, allDataY


def train_data_in(allX, allY, sizeI, batch_size):
    data = sio.loadmat('WV2Data/responseCorefficient')
    C    = data['C']
    _, p2 = C.shape
    p    = np.int(np.sqrt(p2))
    H,W,S   = allX.shape    
    batch_X = np.zeros((batch_size, sizeI, sizeI, S),'f')
    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    sizeZ   = np.int(sizeI/4);
    batch_Z = np.zeros((batch_size, sizeZ, sizeZ, S),'f')
    for i in range(batch_size):    
        px = random.randint(0,H-sizeI)
        py = random.randint(0,W-sizeI)
        subX = allX[px:px+sizeI:1,py:py+sizeI:1,:]
        subY = allY[px:px+sizeI:1,py:py+sizeI:1,:]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subY = np.rot90(subY)
        for j in range(vFlip):
            subX = subX[:,::-1,:]
            subY = subY[:,::-1,:]
        for j in range(hFlip):
            subX = subX[::-1,:,:]
            subY = subY[::-1,:,:]
        batch_X[i,:,:,:] = subX
        batch_Y[i,:,:,:] = subY
## 这里要重写一下
    padnum = np.int((p-4)/2)
    pad_X = np.lib.pad(batch_X, padnum, 'symmetric')
    pad_X = pad_X[padnum:batch_size+padnum,:,:,padnum:padnum+8]
    h = 0;
    for j in range(p):
        for k in range(p):
            batch_Z = batch_Z + C[0,h]*pad_X[:,j:sizeZ*4+j:4,k:sizeZ*4+k:4,:]
            h = h+1;

    return batch_X, batch_Y, batch_Z


