import tensorflow as tf


def conv_net(x,keep_prob,second_inception_layer = False):


    # one layer of inception
    filter1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 96], mean=0, stddev=0.08))
    filter2 = tf.Variable(tf.truncated_normal(shape = [3,3,96,96],mean = 0,stddev  =0.08))

    in_layer1 = tf.nn.conv2d(X, filter1, strides=[1, 1, 1, 1], padding='SAME')
    in_layer2 = tf.nn.conv2d(in_layer1, filter2, strides=[1, 1, 1, 1], padding='SAME')

    in_layer2 = tf.nn.relu(in_layer2)
    inception1 = tf.layers.batch_normalization(in_layer2)

    # 2nd parallel layer of inception
    filter3 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 32], mean=0, stddev=0.08))
    filter4 = tf.Variable(tf.truncated_normal(shape=[5, 5,32, 32], mean=0, stddev=0.08))

    in_layer3 = tf.nn.conv2d(X, filter3, strides=[1, 1, 1, 1], padding='SAME')
    in_layer4 = tf.nn.conv2d(in_layer3, filter4, strides=[1, 1, 1, 1], padding='SAME')

    in_layer4 = tf.nn.relu(in_layer4)
    inception2 = tf.layers.batch_normalization(in_layer4)

    inception2 = tf.layers.dropout(inception2,keep_prob)

    # 3rd parallel layer of inception

    filter5 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 96], mean=0, stddev=0.08))

    in_layer5 = tf.nn.conv2d(X, filter5, strides=[1, 1, 1, 1], padding='SAME')

    in_layer5 = tf.nn.relu(in_layer5)
    inception3 = tf.layers.batch_normalization(in_layer5)

    inception3 = tf.layers.dropout(inception3,keep_prob)

    # 4th parallel layer of max_pooling
    inception4_pool = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    # Applying 1X1 convolution to the pooled version
    filter6 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 32], mean=0, stddev=0.08))

    in_layer6 = tf.nn.conv2d(inception4_pool, filter6, strides=[1, 1, 1, 1], padding='SAME')
    # Relu activation
    in_layer6 = tf.nn.relu(in_layer6)
    # batch normalization
    inception4 = tf.layers.batch_normalization(in_layer6)

    # Concatenation of the result

    inception = keras.layers.concatenate([inception1,inception2,inception3,inception4],axis = 3)
    print(inception.shape)


    '''
    2nd inception layer
    
    following same architecture
    '''
    if second_inception_layer == True:

        filter21 = tf.Variable(tf.truncated_normal(shape = [1,1,256,512],mean = 0,stddev = 0.08))
        filter22 = tf.Variable(tf.truncated_normal(shape = [3,3,512,512],mean = 0,stddev=0.09))

        in2_layer1 = tf.nn.conv2d(inception,filter21,strides = [1,1,1,1],padding = 'SAME')
        in2_layer2 = tf.nn.conv2d(in2_layer1,filter22,strides = [1,1,1,1],padding = 'SAME')

        in2_layer2 = tf.nn.relu(in2_layer2)

        inception21 = tf.layers.batch_normalization(in2_layer2)

        inception21 = tf.layers.dropout(inception21,keep_prob)

        # 2nd parallel layer
        filter23 = tf.Variable(tf.truncated_normal(shape = [1,1,256,256],mean = 0,stddev = 0.08))
        filter24 = tf.Variable(tf.truncated_normal(shape = [5,5,256,256],mean = 0,stddev= 0.08))

        in2_layer3 = tf.nn.conv2d(inception,filter23,strides = [1,1,1,1],padding = 'SAME')
        in2_layer4 = tf.nn.conv2d(in2_layer3,filter24,strides = [1,1,1,1],padding = 'SAME')

        in2_layer4 =  tf.nn.relu(in2_layer4)

        inception22 = tf.layers.batch_normalization(in2_layer4)

        inception22 = tf.layers.dropout(inception22,keep_prob)

        # 3rd parallel layer
        filter25 = tf.Variable(tf.truncated_normal(shape = [1,1,256,256],mean = 0,stddev = 0.08))

        in2_layer5 = tf.nn.conv2d(inception,filter25, strides  =[1,1,1,1],padding = 'SAME')

        in2_layer5 = tf.nn.relu(in2_layer5)

        inception23 = tf.layers.batch_normalization(in2_layer5)

        inception23 = tf.layers.dropout(inception23,keep_prob)

        # 4th parallel avg pooling layer
        inception24_pool = tf.nn.max_pool(inception, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # Applying 1X1 convolution to the pooled version
        filter26 = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 256], mean=0, stddev=0.08))

        in2_layer6 = tf.nn.conv2d(inception24_pool, filter26, strides=[1, 1, 1, 1], padding='SAME')
        # Relu activation
        in2_layer6 = tf.nn.relu(in2_layer6)
        # batch normalization
        inception24 = tf.layers.batch_normalization(in2_layer6)


        # concatenation of the layers
        inception_layer = keras.layers.concatenate([inception21,inception22,inception23,inception24],axis = 3)

        print(inception_layer.shape)

        return inception_layer

    '''
    our dimension is 28X28X1280 
    and our original dataset has dimensions of 28X28X1
    '''

def fully_connected(inception_layers,keep_prob):

    # Moving with general convolutions
    gen_filter1 = tf.Variable(tf.truncated_normal(shape = [3,3,1280,1024],mean = 0,stddev=0.08))
    gen_filter2 = tf.Variable(tf.truncated_normal(shape = [3,3,1024,512],mean = 0,stddev=0.08))
    gen_filter3 = tf.Variable(tf.truncated_normal(shape=[3, 3,512,256], mean=0, stddev=0.08))

    # general convolution layer -1
    gen_conv1 = tf.nn.conv2d(inception_layers,gen_filter1,strides = [1,1,1,1],padding = 'SAME')
    gen_conv1 = tf.nn.relu(gen_conv1)
    pool_conv1 = tf.nn.max_pool(gen_conv1,ksize = [1,2,2,1],strides = [1,2,2,1],padding  = 'SAME')
    conv1 = tf.layers.batch_normalization(pool_conv1)

    # general convolution layer-2
    gen_conv2 = tf.nn.conv2d(conv1, gen_filter2, strides=[1, 1, 1, 1], padding='SAME')
    gen_conv2 = tf.nn.relu(gen_conv2)
    pool_conv2 = tf.nn.max_pool(gen_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.layers.batch_normalization(pool_conv2)
    print(conv2.shape)

    # general convolution layer-3
    gen_conv3 = tf.nn.conv2d(conv2, gen_filter3, strides=[1, 1, 1, 1], padding='SAME')
    gen_conv3 = tf.nn.relu(gen_conv3)
    pool_conv3 = tf.nn.max_pool(gen_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.layers.batch_normalization(pool_conv3)
    print(conv3.shape)


    # fully connected layers
    flat =  tf.contrib.layers.flatten(conv3)
    print(flat.shape)

    # connection 1
    full1 = tf.contrib.layers.fully_connected(inputs = flat,
                                              num_outputs = 1024,
                                              activation_fn = tf.nn.relu)
    full1 = tf.nn.dropout(full1,keep_prob = keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # conncetion 2
    full2 = tf.contrib.layers.fully_connected(inputs=full1,
                                              num_outputs=512,
                                              activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob=keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # connection 3
    full3 = tf.contrib.layers.fully_connected(inputs=full2,
                                              num_outputs=128,
                                              activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob=keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # output layer
    output = tf.contrib.layers.fully_connected(inputs  = full3,
                                               num_outputs =  26,
                                               activation_fn = tf.nn.softmax)

    return output.shape

from keras.datasets import mnist

import keras


(X_train,y_train),(X_test,y_test) = mnist.load_data()

import numpy as np


print(X_train.shape)
# Reshaping the data
X = X_train.reshape(-1,28,28,1)
print(X.shape)

x = X[:100,:]
x = x.astype('float32')
x = x/255.0
X = tf.placeholder(tf.float32,shape = (None,28,28,1))

inception_layer = conv_net(X,0.7,second_inception_layer=True)
print(conv_net(X,0.7,second_inception_layer=True))
print(fully_connected(inception_layers=inception_layer,keep_prob=0.7))