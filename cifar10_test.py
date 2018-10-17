# -*- coding = utf-8 -*-


'''#-------------------------------------------------------------#
20181007
AlexNet


Type:            Maps   Size       Kernelsize   Stride  Padding

input            3      224x224    -            -       -

Conv             96     55x55      11x11        4       SAME
MaxPooling       96     27x27      3x3          2       VALID
Conv             256    27x27      5x5          1       SAME
MaxPooling       256    13x13      3x3          2       VALID

Conv             384    13x13      3x3          1       SAME
Conv             384    13x13      3x3          1       SAME
Conv             256    13x13      3x3          1       SAME

fully connected  -      4096       5x5          1       -
fully connected  -      4096       -            -       -
fully connected  -      1000       -            -       -

#-------------------------------------------------------------#'''


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time
import pickle
import os
import sys



mnist = input_data.read_data_sets('./../data/MNIST_data' , one_hot = True)
#x_image = tf.reshape(mnist_x, [-1,28,28,1])







def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
                datadict = pickle.load(f,encoding='latin1')
                X = datadict['data']
                Y = datadict['labels']
                X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
                Y = np.array(Y)
                return X, Y



def load_CIFAR10(ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        
        for b in range(1,6):
                f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
                X, Y = load_CIFAR_batch(f)
                xs.append(X)
                ys.append(Y)
        
        Xtr = np.concatenate(xs)#使变成行向量
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte





if __name__ == '__main__': 
        a , b = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')
        '''
        print('a is: ' + str(a))
        print('b is: ' + str(b))
        
        
        print('a type is: ' + str(type(a)))
        print('b type is: ' + str(type(b)))

        print('a length is : ' + str(len(a)))
        print('b length is : ' + str(len(b)))

        c , d , e , f = load_CIFAR10('./../data/cifar-10-batches-py')
        print('c is: ' + str(c))
        print('d is: ' + str(d))
        print('e is: ' + str(e))
        print('f is: ' + str(f))


        print('b length is : ' + str(len(b)))
        print('b[0] length is : ' + str(len(b[0])))

        print('a is: ' + str(a))
        '''
        print('a[0] is: '       + str(a[0]))
        # a[0] 和 batch[0] 格式差不多
        print('a[0][0] is: '    + str(a[0][0]))
        print('a[0][0][0] is: ' + str(a[0][0][0]))
        


        print('mnist is :' + str(mnist))
        #print('x_image is :' + str(x_image))

       
        batch = mnist.train.next_batch(64)

        print('batch[0] is :' + str(batch[0]))
        
















