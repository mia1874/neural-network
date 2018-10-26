# -*- coding = utf8 -*-


file_info = (
		'''
		GAN
		Using for MNIST , CIFAR-10
		Author: MrZQ
		20181026


		v1.0.0.1026_alpha Update
				1. GAN working on MNIST and CIFAR10 data set
				2. Bug fix

		''')


import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#import cv2
import time
import pickle
import os
import sys
import cifar10_load
import cifar10_input


#-------------variable init-----------------#
keep_prob = tf.placeholder("float")
learning_rate = 0.001
learning_rate_init = 0.01

is_training = 1
n_in        = 0

batch_size  = 100
accuracy    = []
saver       = []
optimizer_1 = []
loss        = []
predict     = []

data_dir    = './../../data/cifar-10-batches-py'





mnist       = input_data.read_data_sets('./../../data/MNIST_data' , one_hot = True)
mnist_x     = tf.placeholder("float", shape = [None, 784])
mnist_y     = tf.placeholder("float", shape = [None, 10])
