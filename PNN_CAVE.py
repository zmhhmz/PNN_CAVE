import numpy as np
import tensorflow as tf
import os
import random
from utils import interp23, down_img, input_prep
import CAVE_dataReader as Crd
import scipy.io as sio
from model_new import PNN
import re 

param={
    'mode':'train', # train or test
    'epoch':20,
    'batch_iter':2000,
    'lr':0.0001,
    'img_size':96,
    'batch_size':10,
    'train_dir':'train_dir/train_pnnnew3_20epo_res/',
    'data_dir':'CAVEdata/',
    'test_dir':'test_results/test_pnnnew3_20epo_res/',
    'save_model_name':'PNN_model',
    'cost':'L1',
    'residual':True,
    'regol':False,
    'reg_weight':0.000001,
    'ratio':32,
    'gpu':True,
    'tensorboard':True,
    'channel1':3, 
    'channel2':31,
    'padSize':16,
    'NumResNet':30,
    'Target_adaptive':False
}


def train():
    if not os.path.exists(param['train_dir']):
        os.makedirs(param['train_dir'])   
    random.seed( 1 ) 

    img_size = param['img_size']
    ch1 = param['channel1']
    ch2 = param['channel2']

    I = tf.placeholder(tf.float32, shape=(None, img_size,img_size, ch1+ch2)) 
    I_g = tf.placeholder(tf.float32,shape=(None,img_size,img_size,ch2)) 
    I_out,reg = PNN(I,param)
    lr_ = param['lr']
    lr = tf.placeholder(tf.float32 ,shape = [])
    loss1 = tf.reduce_mean(tf.abs(I_out-I_g))
    if param['regol']:
        loss2 = param['reg_weight']*reg
    else:
        loss2=0
    loss = loss1 + loss2
    g_optim =  tf.train.AdamOptimizer(lr).minimize(loss)

    if param['tensorboard']:
        if param['regol']:
            tf.summary.scalar('loss1',loss1)
            tf.summary.scalar('loss2',loss2)
        tf.summary.scalar('loss',loss)
        merged = tf.summary.merge_all() 
        writer = tf.summary.FileWriter('logs',graph=tf.get_default_graph()) 

    saver = tf.train.Saver(max_to_keep = 5)
    save_path = param['train_dir']+param['save_model_name']
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)    
    epoch = param['epoch']

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state(save_path):   
            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d",ckpt)
            if len(ckpt_num)==3:
                start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
            elif len(ckpt_num)==2:
                start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success")
        else:
            print("re-training")
            start_point = 0
        allX, allY = Crd.all_train_data_in()
        for j in range(start_point,epoch):  #start point
            if j+1 >(param['epoch']/3):
                lr_ = param['lr']*0.1
            if j+1 >(2*param['epoch']/3):
                lr_ = param['lr']*0.01

            for num in range(param['batch_iter']):
                print("...Training with the %d-th batch of the %d-th epoch... "%(num+1,j+1))
                batch_X, batch_Y, batch_Z = Crd.train_data_in(allX, allY, param['img_size'], param['batch_size'])
                I_HS_HR = np.transpose(batch_X,(0,3,1,2))
                I_HS = np.transpose(batch_Z,(0,3,1,2))
                I_MS = np.transpose(batch_Y,(0,3,1,2))
                I_input = input_prep(I_HS, I_MS, ratio = param['ratio'])
                I_in = np.transpose(I_input,[0,2,3,1])
                I_ref = np.transpose(I_HS_HR,[0,2,3,1])
                if param['residual']:
                    I_ref=I_ref-I_in[:,:,:,:param['channel2']]
                if param['regol']:
                    _, lossvalue,lossvalue1,lossvalue2 = sess.run([g_optim,loss,loss1,loss2],{I:I_in,I_g:I_ref,lr:lr_})
                    print("loss: {0}, loss1: {1}, loss2: {2}".format(lossvalue,lossvalue1,lossvalue2))
                else:
                    _, lossvalue = sess.run([g_optim,loss],{I:I_in,I_g:I_ref,lr:lr_})
                    print("loss: {0}".format(lossvalue))

                if param['tensorboard'] and num%100==99:
                    result = sess.run(merged,feed_dict={I:I_in,I_g:I_ref,lr:lr_})
                    writer.add_summary(result,num//100 + param['batch_iter']*j//100) #将日志数据写入文件
            saver.save(sess, save_path, global_step = j+1)
            ckpt = tf.train.latest_checkpoint(param['train_dir'])
            saver.restore(sess, ckpt)

def testAll():
    if not os.path.exists(param['test_dir']):
        os.makedirs(param['test_dir'])

    I = tf.placeholder(tf.float32, shape=(None, 512,512, 34))
    I_out,_ = PNN(I)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)

    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(param['train_dir'])
        saver.restore(sess, ckpt) 
        
        files =os.listdir('CAVEdata/X/')
        files.sort()
        for i in range(32):
            I_HS,I_MS = Crd.generate_test_data2(param['ratio'],files[i])
            I_in = input_prep(I_HS, I_MS, ratio = param['ratio'])
            I_in = np.transpose(I_in,[0,2,3,1])
            
            I_pred = sess.run([I_out],{I:I_in})
        
            if param['residual']:
                I_res = I_in[:,:,:,:param['channel2']]
                I_pred = np.squeeze(I_pred) + np.squeeze(I_res)
            else:
                I_pred = np.squeeze(I_pred)

            sio.savemat(param['test_dir']+files[i], {'outX': I_pred})     
            print(files[i] + ' done!')



if __name__ == '__main__':
    if param['gpu']:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
    with tf.device(dev):
        if param['mode']=='train':
            train()
        else:
            testAll()

