import sys
import scipy.io as sio
import os
import numpy as np
import scipy.misc as misc
from scipy import ndimage
# import theano
# import theano.tensor as T

def interp23(image, ratio):
    if (2**np.log2(ratio).round() != ratio):
        print('Error: only resize factors of power 2')
        return

    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image=I1LRU
        
    return image

def saveLayer(layer, model):
    """add trained layer to model
        layer: list of layers after training
        model: PNN_model
    """
    padsize=0
    layers=[]
    for l in layer:
        w,b = [np.asarray(l.w.eval()), np.asarray(l.b.eval())]
        layers.append(w)
        layers.append(b)
        padsize+=l.w.eval().shape[2]-1
    model['padSize']=padsize
    model['layers']=layers
    return model

def down_img(I_HS, ratio):  
    I_HS_LP = np.zeros((I_HS.shape[0],I_HS.shape[1],int(np.round(I_HS.shape[2]/ratio)+4),int(np.round(I_HS.shape[3]/ratio)+4)))    
    I_HS_LR = np.zeros((I_HS.shape[0],I_HS.shape[1],int(np.round(I_HS.shape[2]/ratio)),int(np.round(I_HS.shape[3]/ratio))))    
    for i in range(I_HS.shape[0]):
        for idim in range(I_HS.shape[1]):
            imslp_pad=np.pad(I_HS[i,idim,:,:],int(2*ratio),'symmetric')
            I_HS_LP[i,idim,:,:]=misc.imresize(imslp_pad,1/ratio,'bicubic',mode='F')
        I_HS_LR[i,:,:,:] = I_HS_LP[i,:,2:-2,2:-2]
    return I_HS_LR

def input_prep(I_HS_LR,I_MS,ratio):
    I_in = np.zeros((I_HS_LR.shape[0],I_HS_LR.shape[1]+I_MS.shape[1],I_MS.shape[2],I_MS.shape[3]))
    for i in range(I_HS_LR.shape[0]):
        I_HS_i = interp23(I_HS_LR[i], ratio)
        I_in_i = np.vstack((I_HS_i, I_MS[i])).astype('single')
        I_in[i,:,:,:]=I_in_i
    return I_in
