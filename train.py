import CAVE_dataReader as Crd
import theano
import theano.tensor as T
import numpy as np
import os
import scipy.io as sio
import scipy.misc as misc
from utils import interp23, down_img, input_prep
from model import Network, ConvLayer

theano.config.floatX= 'float32'

param={
    'rand':False, # True for using randomly initialized parameters (likely to explode);
                  # False for using pretrained parameters
    'total_epoch':2,
    'in_epoch':6,
    'batch_iter':2,
    'lr':0.0001,
    'img_size':96,
    'batch_size':10,
    'train_dir':'train_dir/',
    'save_model_name':'PNN_model2.mat',
    'cost':'L1',
    'residual':True,
    'regol':True,
    'ratio':32,
    'gpu':False,
    'channel1':34, 
    'channel2':31
}

# if param['gpu']:
#     os.environ["THEANO_FLAGS"]='floatX=float32,init_gpu_device=cuda0'
# else:
#     os.environ["THEANO_FLAGS"] = "floatX=float32"


if not os.path.exists(param['train_dir']):
    os.makedirs(param['train_dir'])

def train():

    layers=[]
    if param['rand']:
        w1 = np.random.normal(0.0,0.1,(48, param['channel1'], 9, 9))
        b1 = np.random.normal(0.0,0.1,48)
        w2 = np.random.normal(0.0,0.1,(32, 48, 5, 5))
        b2 = np.random.normal(0.0,0.1,32)
        w3 = np.random.normal(0.0,0.1,(param['channel2'], 32, 5, 5))
        b3 = np.random.normal(0.0,0.1,param['channel2'])
        layers.append(ConvLayer(w1,b1))
        layers.append(ConvLayer(w2,b2))
        layers.append(ConvLayer(w3,b3))
    else:
        out = Crd.init_layers()
        for i in range(0,len(out),2):
            layers.append(ConvLayer(out[i],out[i+1]))



    net = Network(layers)

    allX, allY = Crd.all_train_data_in()

    lr_ini = param['lr']

    save_dir = param['train_dir']+param['save_model_name']

    for j in range(0,param['total_epoch']):  #start point
        if j+1 >(param['total_epoch']/3):
            param['lr'] = lr_ini*0.1
        if j+1 >(2*param['total_epoch']/3):
            param['lr'] = lr_ini*0.01

        for num in range(param['batch_iter']):
            print("...Training with the %d-th batch of the %d-th total_epoch... "%(num+1,j+1))
            batch_X, batch_Y, _ = Crd.train_data_in(allX, allY, param['img_size'], param['batch_size'])
            I_HS_HR = np.transpose(batch_X,(0,3,1,2))
            I_HS = down_img(I_HS_HR, ratio = param['ratio'])
            I_MS = np.transpose(batch_Y,(0,3,1,2))
            I_input = input_prep(I_HS, I_MS, ratio = param['ratio'])

            I_in=theano.shared(np.asarray((I_input),dtype=theano.config.floatX))
            I_ref=theano.shared(np.asarray((I_HS_HR),dtype=theano.config.floatX))

            delta = param['channel1'] - param['channel2']
            if param['residual']:
                I_ref=I_ref-I_in[:,:-delta,:,:]
            
            mod = net.SGD(I_in,I_ref,param)
            
        sio.savemat(save_dir,mod)
    print("finished training network")
            
if __name__ == '__main__':
    if param['gpu']:
        os.environ["THEANO_FLAGS"]='init_gpu_device=cuda0'

    train()

