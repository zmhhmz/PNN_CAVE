
import tensorflow as tf

#original version with batchnorm
def PNN(I_in,param,filters=[96,64,31]):
    w1 = create_kernel('w1', [9, 9, 34, filters[0]])
    b1 = tf.Variable(tf.constant(0.0, shape=[filters[0]], dtype=tf.float32), trainable=True, name='b1')
    scale = tf.Variable(tf.ones(filters[0])/20, trainable=True, name=('scale1'))
    beta = tf.Variable(tf.zeros(filters[0]), trainable=True, name=('beta1'))
    conv = tf.nn.conv2d(I_in, w1, [1, 1, 1, 1], padding='SAME')
    feature = tf.nn.bias_add(conv, b1)
    mean, var  = tf.nn.moments(feature,[0, 1, 2])
    feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
    feature_relu = tf.nn.relu(feature_normal)

    w2 = create_kernel('w2', [5, 5, filters[0], filters[1]])
    b2 = tf.Variable(tf.constant(0.0, shape=[filters[1]], dtype=tf.float32), trainable=True, name='b2')
    scale = tf.Variable(tf.ones(filters[1])/20, trainable=True, name=('scale2'))
    beta = tf.Variable(tf.zeros(filters[1]), trainable=True, name=('beta2'))
    conv = tf.nn.conv2d(feature_relu, w2, [1, 1, 1, 1], padding='SAME')
    feature = tf.nn.bias_add(conv, b2)
    mean, var  = tf.nn.moments(feature,[0, 1, 2])
    feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
    feature_relu = tf.nn.relu(feature_normal)
    
    w3 = create_kernel('w3', [5, 5, filters[1], filters[2]])
    b3 = tf.Variable(tf.constant(0.0, shape=[filters[2]], dtype=tf.float32), trainable=True, name='b3')
    scale = tf.Variable(tf.ones(filters[2])/20, trainable=True, name=('scale3'))
    beta = tf.Variable(tf.zeros(filters[2]), trainable=True, name=('beta3'))
    conv = tf.nn.conv2d(feature_relu, w3, [1, 1, 1, 1], padding='SAME')
    feature = tf.nn.bias_add(conv, b3)
    mean, var  = tf.nn.moments(feature,[0, 1, 2])
    x = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
    return x

#deeper version
def PNN2(I_in):
    I_in = resCNNnet('ResBlock',I_in,1,34,30)
    filters=[34,64,64,31]
    last =False
    for i in range(len(filters)-1):
        if i==len(filters)-2:
            last=True
        w = create_kernel('w{0}'.format(i+1), [5, 5, filters[i], filters[i+1]])
        b = tf.Variable(tf.constant(0.0, shape=[filters[i+1]], dtype=tf.float32), trainable=True, name='b{0}'.format(i+1))
        scale = tf.Variable(tf.ones(filters[i+1])/20, trainable=True, name=('scale{0}'.format(i+1)))
        beta = tf.Variable(tf.zeros(filters[i+1]), trainable=True, name=('beta{0}'.format(i+1)))
        conv = tf.nn.conv2d(I_in, w, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, b)
        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        I_in = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        if not last:
            I_in = tf.nn.relu(I_in)
    
    return I_in

def create_kernel(name, shape, initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.1)):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)
    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer, trainable=True)
    return new_variables

def resCNNnet(name,X,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)                                
        return X

    
def resLevel(name, Fsize,X,Channel):
    with tf.variable_scope(name):
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
        scale = tf.Variable(tf.ones([Channel+3])/100, trainable=True, name=('scale1'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 我又加了一层
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[ Channel+3], dtype=tf.float32), trainable=True, name='biases2')
        scale = tf.Variable(tf.ones([ Channel+3])/100, trainable=True, name=('scale2'))
        beta = tf.Variable(tf.zeros([ Channel+3]), trainable=True, name=('beta2'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)
        #
        
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel], dtype=tf.float32), trainable=True, name='biases3')
        scale = tf.Variable(tf.ones([Channel])/100, trainable=True, name=('scale3'))
        beta = tf.Variable(tf.zeros([Channel]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_relu)  #  shortcut  
        return X    
