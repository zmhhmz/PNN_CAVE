# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 22:20:12 2018

@author: XieQi
"""

import os
import numpy as np
import scipy.io as sio  
import random 
from utils import down_img

datapath = 'ChikuseiData/'

def all_train_data_in():
    data = sio.loadmat('ChikuseiData/train/X')
    allDataX = data['chikusei']
    data = sio.loadmat('ChikuseiData/V')
    V    = data['V']
    allDataX = np.tensordot(allDataX,V,(2,0))
    data = sio.loadmat('ChikuseiData/train/Y')
    allDataY = data['Y']
    return allDataX, allDataY


def train_data_in(allX, allY, sizeI, batch_size):
    H,W,S   = allX.shape    
    batch_X = np.zeros((batch_size, sizeI, sizeI, S),'f')
    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    batch_Z = np.zeros((batch_size, 3, 3, S),'f')
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
    for j in range(32):
        for k in range(32):
            batch_Z = batch_Z + batch_X[:,j:512:32,k:512:32,:]/32/32
    return batch_X, batch_Y, batch_Z


def eval_data_in( sizeI=96, batch_size=200):
    data = sio.loadmat('ChikuseiData/eval/X')
    allX = data['chikusei']
    data = sio.loadmat('ChikuseiData/V')
    V    = data['V']
    allX = np.tensordot(allX,V,(2,0))
    data = sio.loadmat('ChikuseiData/eval/Y')
    allY = data['Y']
    H,W,S   = allX.shape    
    batch_X = np.zeros((batch_size, sizeI, sizeI, S),'f')
    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    batch_Z = np.zeros((batch_size, 3, 3, S),'f')
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
    for j in range(32):
        for k in range(32):
            batch_Z = batch_Z + batch_X[:,j:512:32,k:512:32,:]/32/32
    return batch_X, batch_Y, batch_Z

def generate_test_data(test_data_name):
    data = sio.loadmat(datapath+'test/'+test_data_name+'/Y')
    I_MS = np.expand_dims(data['Y'],axis=0)
    data = sio.loadmat(datapath+'test/'+test_data_name+'/Z')
    inZ = data['Z']
    data = sio.loadmat('ChikuseiData/V')
    V    = np.float32(data['V'])
    inZ  = np.tensordot(inZ, V, ([2],[0]))
    I_HS  = np.expand_dims(inZ, axis = 0)
    I_MS = np.transpose(I_MS,(0,3,1,2))
    I_HS = np.transpose(I_HS,(0,3,1,2))
    return I_HS,I_MS
