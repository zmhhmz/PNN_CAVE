import tensorflow as tf

def PNN(I_in,param):
    I_in,reg1 = resCNNnet2('ResBlock',I_in,1,param['channel1']+param['channel2'],52)
    filters=[param['channel1']+param['channel2'],param['channel2']]
    last =False
    reg2=0
    for i in range(len(filters)-1):
        if i==len(filters)-2:
            last=True
        w = create_kernel('w{0}'.format(i+1), [5, 5, filters[i], filters[i+1]])
        reg2i = tf.reduce_sum(w**2)
        reg2=reg2+reg2i
        b = tf.get_variable(name='b{0}'.format(i+1),shape=[filters[i+1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale{0}'.format(i+1),shape=[filters[i+1]],dtype=tf.float32,initializer=tf.constant_initializer(1.0/20),trainable=True)
        beta = tf.get_variable(name='beta{0}'.format(i+1),shape=[filters[i+1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        conv = tf.nn.conv2d(I_in, w, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, b)
        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        I_in = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        if not last:
            I_in = tf.nn.relu(I_in)
    reg = reg1+reg2
    return I_in,reg

def create_kernel(name, shape, initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.1)):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)
    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer, trainable=True)
    return new_variables

def resCNNnet(name,X,j,channel,levelN):
    reg=0
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X,reg_i = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)
            reg = reg+reg_i                               
    return X , reg

def resLevel(name, Fsize,X,Channel): #3层
    with tf.variable_scope(name):
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.get_variable(name='biases1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        beta = tf.get_variable(name='beta1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        reg1 = tf.reduce_sum(kernel**2)
        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 我又加了一层
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.get_variable(name='biases2',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale2',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        beta = tf.get_variable(name='beta2',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        reg2 = tf.reduce_sum(kernel**2)
        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)
        #
        
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel])
        biases = tf.get_variable(name='biases3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        beta = tf.get_variable(name='beta3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        reg3 = tf.reduce_sum(kernel**2)
        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_relu)  #  shortcut  
        reg = reg1+reg2+reg3
        return X,reg

def resCNNnet2(name,X,j,channel,levelN):
    reg=0
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X,reg_i = resLevel2(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)
            reg = reg+reg_i                               
    return X , reg
    
def resLevel2(name, Fsize,X,Channel): #2层
    with tf.variable_scope(name):
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.get_variable(name='biases1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        beta = tf.get_variable(name='beta1',shape=[Channel+3],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        reg1 = tf.reduce_sum(kernel**2)
        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 我又加了一层
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel])
        biases = tf.get_variable(name='biases2',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        scale = tf.get_variable(name='scale2',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        beta = tf.get_variable(name='beta2',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        reg2 = tf.reduce_sum(kernel**2)
        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)
        #
        
        # kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel])
        # biases = tf.get_variable(name='biases3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        # scale = tf.get_variable(name='scale3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(1.0/100),trainable=True)
        # beta = tf.get_variable(name='beta3',shape=[Channel],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)
        # reg3 = tf.reduce_sum(kernel**2)
        # conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        # feature = tf.nn.bias_add(conv, biases)

        # mean, var  = tf.nn.moments(feature,[0, 1, 2])
        # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        # feature_relu = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_relu)  #  shortcut  
        reg = reg1+reg2
        return X,reg
