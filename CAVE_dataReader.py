# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:13:53 2018

@author: XieQi
"""
import h5py
import os
import numpy as np
import scipy.io as sio  
#import MyLib as ML
import random 
#import cv2

datapath = 'CAVEdata/'

def all_train_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat(datapath+'List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk(datapath+'X/'):
           for j in range(20):
#                print(Ind[0,j])
                i = Ind[0,j]-1
                data = sio.loadmat(datapath+'X/'+files[i])
                inX  = data['msi']
#                print(type(inX[1,1,1]))
                allDataX.append(inX)
                data = sio.loadmat(datapath+'Y/'+files[i])
                inY  = data['RGB']
                allDataY.append(inY)
                
    return allDataX, allDataY


def all_test_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat(datapath+'List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk(datapath+'X/'):
           for j in range(12):
#                print(Ind[0,j])
                i = Ind[0,j+20]-1
#                print(i)
                data = sio.loadmat(datapath+'X/'+files[i])
                inX  = data['msi']
                allDataX.append(inX)
                data = sio.loadmat(datapath+'Y/'+files[i])
                inY  = data['RGB']
                allDataY.append(inY)
    return allDataX, allDataY

def train_data_in(allX, allY, sizeI, batch_size, channel=31,dataNum = 20,ratio=32):

    batch_X = np.zeros((batch_size, sizeI, sizeI, channel),'f')
    batch_Y = np.zeros((batch_size, sizeI, sizeI, 3),'f')
    batch_Z = np.zeros((batch_size, sizeI//ratio, sizeI//ratio, channel),'f')

    for i in range(batch_size):

        #crop into sizeI x sizeI
        ind = random.randint(0, dataNum-1)
        X = allX[ind]
        Y = allY[ind]
        px = random.randint(0,512-sizeI)
        py = random.randint(0,512-sizeI)
        subX = X[px:px+sizeI:1,py:py+sizeI:1,:]
        subY = Y[px:px+sizeI:1,py:py+sizeI:1,:]

        #rotate and flip
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

    for j in range(ratio):
        for k in range(ratio):
            batch_Z = batch_Z + batch_X[:,j:512:ratio,k:512:ratio,:]/ratio/ratio

    return batch_X, batch_Y, batch_Z


def eval_data_in():
    allX, allY = all_test_data_in()
    eval_X, eval_Y, eval_Z = train_data_in(allX, allY, 96, 10, 31,12)

    return eval_X, eval_Y, eval_Z

def read_data(file):
    with h5py.File(file, 'r') as hf:
        X = hf.get('X')
        Y = hf.get('Y')
        Z = hf.get('Z')
        return np.array(X), np.array(Y), np.array(Z)

def init_layers(layers_dir = datapath+'layers_init.mat'):
    mat = sio.loadmat(layers_dir)
    layers = mat['layers'][0]
    out=[]
    for i in range(len(layers)):
        out.append(np.squeeze(layers[i]))
    return out
    
