# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:13:53 2018

@author: XieQi
"""
import h5py
import os
import numpy as np
import scipy.io as sio  
from utils import down_img
import random 


datapath = 'CAVEdata/'

def all_train_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat(datapath+'List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    files = os.listdir(datapath+'X/')
    files.sort()
    for j in range(20):
        i = Ind[0,j]-1
        data = sio.loadmat(datapath+'X/'+files[i])
        inX  = data['msi']
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
    files = os.listdir(datapath+'X/')
    files.sort()
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

def get_layers(layers_dir = datapath+'layers_init.mat'):
    mat = sio.loadmat(layers_dir)
    layers = mat['layers'][0]
    out=[]
    for i in range(len(layers)):
        out.append(np.squeeze(layers[i]))
    return out
    
def generate_test_data(ratio,test_data_name):
    #test_data_name = 'chart_and_stuffed_toy_ms'
    data = sio.loadmat(datapath+'Y/'+test_data_name)
    I_MS = data['RGB']
    data = sio.loadmat(datapath+'X/'+test_data_name)
    I_HS_HR = data['msi']

    I_MS = np.expand_dims(I_MS, axis = 0)
    I_MS = np.transpose(I_MS,(0,3,1,2))
    I_HS_HR = np.expand_dims(I_HS_HR, axis = 0)
    I_HS_HR = np.transpose(I_HS_HR,(0,3,1,2))
    I_HS = down_img(I_HS_HR, ratio)

    return I_HS,I_MS

def generate_test_data2(ratio,test_data_name):
    data = sio.loadmat(datapath+'Y/'+test_data_name)
    I_MS = data['RGB']
    data = sio.loadmat(datapath+'Z/'+test_data_name)
    I_HS  = data['Zmsi']

    I_MS = np.expand_dims(I_MS, axis = 0)
    I_MS = np.transpose(I_MS,(0,3,1,2))
    I_HS = np.expand_dims(I_HS, axis = 0)
    I_HS = np.transpose(I_HS,(0,3,1,2))

    return I_HS,I_MS