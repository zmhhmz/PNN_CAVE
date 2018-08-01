import CAVE_dataReader as Crd
import theano
import theano.tensor as T
import numpy as np
import os
import scipy.io as sio
import scipy.misc as misc
from utils import interp23, down_img, input_prep
from model import Network, ConvLayer

param={
    'mode':'train', # train or test
    'rand':False, # True for using randomly initialized parameters (likely to explode);
                  # False for using pretrained parameters 
    'total_epoch':20,
    'in_epoch':10,
    'batch_iter':1000,
    'lr':0.0001,
    'img_size':96,
    'batch_size':10,
    'train_dir':'train_dir/',
    'data_dir':'CAVEdata/',
    'test_dir':'test_results/',
    'save_model_name':'PNN_model2.mat',
    'cost':'L1',
    'residual':True,
    'regol':True,
    'ratio':32,
    'gpu':True,
    'channel1':34, 
    'channel2':31,
    'padSize':16
}

theano.config.floatX= 'float32'
if param['gpu']:
    THEANO_FLAGS='device=cuda0,init_gpu_device=cuda0'

    #test gpu
    import time
    vlen = 10 * 30 * 768  
    iters = 1000
    rng = np.random.RandomState(22)
    x = theano.shared(np.asarray(rng.rand(vlen), theano.config.floatX))
    f = theano.function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    #print("Looping %d times took %f seconds" % (iters, t1 - t0))
    #print("Result is %s" % (r,))
    if np.any([isinstance(x.op, T.Elemwise) and
                ('Gpu' not in type(x.op).__name__)
                for x in f.maker.fgraph.toposort()]):
        print('Using the cpu')
    else:
        print('Using the gpu')


def train():
    if not os.path.exists(param['train_dir']):
        os.makedirs(param['train_dir'])

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
        out = Crd.get_layers()
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

            
            if param['residual']:
                I_ref=I_ref-I_in[:,:param['channel2'],:,:]
            
            mod = net.SGD(I_in,I_ref,param)
            
        sio.savemat(save_dir,mod)
    print("finished training network")

def pnn_test(I_HS,I_MS,model):
    layers=[]
    out = Crd.get_layers(model['train_dir']+model['save_model_name'])
    for i in range(0,len(out),2):
        layers.append(ConvLayer(out[i],out[i+1]))
    net = Network(layers)
    I_in = input_prep(I_HS, I_MS, ratio = model['ratio'])
    if model['residual']:
        I_res = I_in[:,:param['channel2'],:,:]
    I_in = np.expand_dims(np.pad(np.squeeze(I_in), ((0,0),(model['padSize']//2,model['padSize']//2),(model['padSize']//2,model['padSize']//2)),mode='edge'),axis=0)
    if model['residual']:
        I_out=net.build(I_in)+I_res[:,:model['channel2'],:,:]
    else:
        I_out=net.build(I_in)
    return I_out

def test(model): #test all data
    if not os.path.exists(model['test_dir']):
        os.makedirs(model['test_dir'])
    for root, dirs, files in os.walk(model['data_dir']+'X/'):
        for i in range(len(files)):
            I_HS,I_MS = Crd.generate_test_data(model['ratio'],files[i])
            I_out = pnn_test(I_HS,I_MS,model)
            sio.savemat(model['test_dir']+files[i],{'outX':I_out})

            
if __name__ == '__main__':

    if param['mode']=='train':
        train()
    else:
        test(param)

