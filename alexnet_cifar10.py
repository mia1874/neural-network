#-*- coding : utf-8 -*-

'''#------------------------network info---------------------------------#
20181008
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

#---------------------------------------------------------------------#'''

file_info = (
		'''
		AlexNet
		Using for MNIST , CIFAR-10
		Author: MrZQ
		20181008


		v1.0.0.1003_alpha Update
				1. AlexNet working on MNIST data set
				2. Bug fix-

		v1.0.1.1008_alpha Update
				1. Working well on MNIST data set
				2. Fix bug for 'loss is Nan'
				3. ready for adding cifar10 data set

		v1.0.2.1012_alpha Update
				1. Bug fix and running no error
				2. Origin
					accuracy ----> 10% ~ 20%
				3. Remove LRN
					loss drop very low (learning_rate ----> 0.001)
					Iter 2944, Minibatch Loss = 85672786395136.000000, Training Accuracy = 0.15625
				4. Dynamic update learning_rate
					0.1 ---> 0.01 ---> 0.001
					accuracy ----> 10% ~ 28%
				5. Dropout
					keep_prob

				6. Hyperparameter
					hyperparameter need to change, 11x11 is too big for 32x32 pic>

				7. Batch size 
					smaller batch size

				8. Batch normalization
					

		''')


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import pickle
import os
import sys


#-------------variable init-----------------#
keep_prob = tf.placeholder("float")
learning_rate = 0.001
learning_rate_init = 0.01

batch_size = 64
accuracy = []
saver = []
optimizer_1 = []
loss = []

#hyperparameter for MNIST
mnist_weights = {
		'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
		'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
		'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
		'wd2': tf.Variable(tf.random_normal([4096, 4096])),
		'out': tf.Variable(tf.random_normal([4096, 10]))
}
mnist_biases = {
		'bc1': tf.Variable(tf.random_normal([64])),
		'bc2': tf.Variable(tf.random_normal([192])),
		'bc3': tf.Variable(tf.random_normal([384])),
		'bc4': tf.Variable(tf.random_normal([384])),
		'bc5': tf.Variable(tf.random_normal([256])),
		'bd1': tf.Variable(tf.random_normal([4096])),
		'bd2': tf.Variable(tf.random_normal([4096])),
		'out': tf.Variable(tf.random_normal([10]))
}


'''

cifar10_weights = {
		'wc1': tf.Variable(tf.random_normal([11, 11, 3, 64])),
		'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
		'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
		#'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
		'wd2': tf.Variable(tf.random_normal([4096, 4096])),
		'out': tf.Variable(tf.random_normal([4096, 10]))
}
cifar10_biases = {
		'bc1': tf.Variable(tf.random_normal([64])),
		'bc2': tf.Variable(tf.random_normal([192])),
		'bc3': tf.Variable(tf.random_normal([384])),
		'bc4': tf.Variable(tf.random_normal([384])),
		'bc5': tf.Variable(tf.random_normal([256])),
		'bd1': tf.Variable(tf.random_normal([4096])),
		'bd2': tf.Variable(tf.random_normal([4096])),
		'out': tf.Variable(tf.random_normal([10]))
}









#hyperparameter for Cifar10
cifar10_weights = {
		'wc1': tf.Variable(tf.random_normal([3, 3, 3, 24])),
		'wc2': tf.Variable(tf.random_normal([3, 3, 24, 96])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 96, 192])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 192, 192])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 192, 96])),
		'wd1': tf.Variable(tf.random_normal([96*24*24, 1024])),
		#'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
		'wd2': tf.Variable(tf.random_normal([1024, 1024])),
		'out': tf.Variable(tf.random_normal([1024, 10]))
}
cifar10_biases = {
		'bc1': tf.Variable(tf.random_normal([24])),
		'bc2': tf.Variable(tf.random_normal([96])),
		'bc3': tf.Variable(tf.random_normal([192])),
		'bc4': tf.Variable(tf.random_normal([192])),
		'bc5': tf.Variable(tf.random_normal([96])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		'bd2': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([10]))
}

'''

cifar10_weights = {
		'wc1': tf.Variable(tf.random_normal([11, 11, 3, 64])),
		'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
		'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
		#'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
		'wd2': tf.Variable(tf.random_normal([4096, 4096])),
		'out': tf.Variable(tf.random_normal([4096, 10]))
}
cifar10_biases = {
		'bc1': tf.Variable(tf.random_normal([64])),
		'bc2': tf.Variable(tf.random_normal([192])),
		'bc3': tf.Variable(tf.random_normal([384])),
		'bc4': tf.Variable(tf.random_normal([384])),
		'bc5': tf.Variable(tf.random_normal([256])),
		'bd1': tf.Variable(tf.random_normal([4096])),
		'bd2': tf.Variable(tf.random_normal([4096])),
		'out': tf.Variable(tf.random_normal([10]))
}


#-------------data set import-----------------#

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



def batch_next(train_data, train_label, batch_size_1):
		# 传入的cifar10的train_label格式为 list [6 9 9 ... 9 1 1]
		index = [ i for i in range(0,len(train_label)) ]
		np.random.shuffle(index);
		batch_data  = []
		batch_label = []
		for j in range(0,batch_size_1):


			train_label = list(train_label)


			#随机产生图片序列？
			batch_data.append(train_data[index[j]]);
			#print('\n----------------------------------------batch label before before is:' +  str(batch_label))

			#转换return的格式？
			# batch_label的格式是 int
			# 所需格式是10元素list
			# 需要1 --->[1 0 0 0 0 0 0 0 0 0]

			#print('\n----------------------------------------batch data is:' +  str(batch_data))
			'''
			#调试新建batch函数的log , 查看生成的batch是否正确

			print('\n----------------------------------------batch label is:' +  str(batch_label))
			print('\n----------------------------------------batch label type is:' +  str(type(batch_label)))

			print('\n----------------------------------------batch label [0] is:' +  str(batch_label[0]))
			'''

			label_temp = figure_2_label(train_label[j])

			'''
			print('\n----------------------------------------batch label is:' +  str(batch_label))
			print('\n----------------------------------------batch label type is:' +  str(type(batch_label)))
			'''


			batch_label.append(label_temp)


		batch_data  = np.array(batch_data)
		batch_label = np.array(batch_label)
		#print('\n----------------------------------------batch label before is:' +  str(batch_label))

		batch_data  = batch_data.reshape([-1,3072])

		return batch_data , batch_label



def label_2_figure(x):
		if x   == '[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]':
				return('0')

		elif x == '[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]':
				return('1')

		elif x == '[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]':
				return('2')

		elif x == '[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]':
				return('3')

		elif x == '[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]':
				return('4')

		elif x == '[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]':
				return('5')

		elif x == '[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]':
				return('6')

		elif x == '[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]':
				return('7')

		elif x == '[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]':
				return('8')

		elif x == '[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]':
				return('9')

		else:
				return('Error')



def figure_2_label(x):
		x = str(x)
		if x   == '0' :
				return(np.array([1. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]))

		elif x == '1' :
				return(np.array([0. ,1. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]))

		elif x == '2' :
				return(np.array([0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]))

		elif x == '3' :
				return(np.array([0. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0.]))

		elif x == '4' :
				return(np.array([0. ,0. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0.]))

		elif x == '5' :
				return(np.array([0. ,0. ,0. ,0. ,0. ,1. ,0. ,0. ,0. ,0.]))

		elif x == '6' :
				return(np.array([0. ,0. ,0. ,0. ,0. ,0. ,1. ,0. ,0. ,0.]))

		elif x == '7' :
				return(np.array([0. ,0. ,0. ,0. ,0. ,0. ,0. ,1. ,0. ,0.]))

		elif x == '8' :
				return(np.array([0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,1. ,0.]))

		elif x == '9' :
				return(np.array([0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,1.]))

		else:
				return('Error')




# mnist shape 28 x 28
mnist       = input_data.read_data_sets('./../data/MNIST_data' , one_hot = True)
mnist_x     = tf.placeholder("float", shape = [None, 784])
mnist_y     = tf.placeholder("float", shape = [None, 10])



# cifar10 shape 32 x 32 x 3
# cifar10   = input_data.read_data_sets('./../data/cifar-10-batches-py' , one_hot = True)
# input_data.read_data_sets只能读取mnist？
# from tensorflow.examples.tutorials.mnist import input_data
# 从mnist import input_data

#cifar10     = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')
cifar10     = load_CIFAR10('./../data/cifar-10-batches-py')

#cifar10_x   = tf.placeholder("float", shape = [None, 32 ,32 ,3])
cifar10_x   = tf.placeholder("float", shape = [None, 3072])
cifar10_y   = tf.placeholder("float", shape = [None, 10])


print('cifar10  is: ' + str(cifar10))
print('mnist[0] is: ' + str(mnist[0]))
print('cifar10[1] is' + str(cifar10[1]))
print('cifar10[1][0] is ' + str(cifar10[1][0]))
print('cifar10[1][1] is ' + str(cifar10[1][1]))
print('cifar10[1][2] is ' + str(cifar10[1][2]))

#-------------function init-----------------#
def weight_variable(shape):
		initial = tf.truncated_normal(shape , stddev=0.1)
		return tf.Variable(initial)

def bias_variable(shape):
		initial = tf.constant(0.1 , shape = shape)
		return tf.Variable(initial)

def conv2d_p(x ,W ,s):
		return tf.nn.conv2d(x, W, strides=[1,s,s,1] , padding = 'SAME')

def conv2d_1(x ,W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1] , padding = 'SAME')

def max_pool_3x3(x):
		return tf.nn.max_pool(x,  ksize=[1,2,2,1] , strides=[1,2,2,1] , padding = "SAME")

def norm(x):
		return tf.nn.lrn(x , 4 , bias = 1.0 , alpha = 0.001/9.0 , beta = 0.75)





def dropout(x):
		return(tf.nn.dropout(x , keep_prob))

def softmax(x, W, b):
		return(tf.nn.softmax(tf.matmul(x, W) + b))

def loss_cross_entropy(x, y):
		return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x , labels = y)))




def optimizer(learning_rate, cross_entropy):
		return(tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy))

def correct_predict(a,b):
		return(tf.equal(tf.argmax(a,1) , tf.argmax(b,1)))

def accuracy(predict):
		return(tf.reduce_mean(tf.cast(predict, "float")))




#-------------working flow-----------------#
'''
Layer 0:    Input
Layer 1:    Conv + Max pooling
Layer 2:    Conv + Max pooling
Layer 3:    Conv + Conv + Conv
Layer 4:    Fully connected + Fully connected
Layer 5:    Fully connected
Layer 6:    Output
'''

def working_flow_mnist():
		global accuracy
		global saver
		global optimizer_1
		global loss

		#layer 0
		x_image = tf.reshape(mnist_x, [-1,28,28,1])


		#layer 1
		h_conv1 = tf.nn.relu(conv2d_1(x_image, mnist_weights['wc1']) + mnist_biases['bc1'])


		#print('\n*******h_conv1 is: ' + str(h_conv1))
		#print('\n*******h_conv1 type is: ' + str(type(h_conv1)))
		h_pool1 = max_pool_3x3(h_conv1)


		#print('\n*******h_pool1 is: ' + str(h_pool1))
		#print('\n*******h_pool1 type is: ' + str(type(h_pool1)))


		h_norm1 = norm(h_pool1)

		#layer 2
		h_conv2 = tf.nn.relu(conv2d_1(h_norm1, mnist_weights['wc2'] ) + mnist_biases['bc2'])

		#print('\n*******h_conv2 is: ' + str(h_conv2))
		#print('\n*******h_conv2 type is: ' + str(type(h_conv2)))
		h_pool2 = max_pool_3x3(h_conv2)

		#print('\n*******h_pool2 is: ' + str(h_pool2))
		#print('\n*******h_pool2 type is: ' + str(type(h_pool2)))

		h_norm2 = norm(h_pool2)

		############################
		#layer 3
		h_conv3 = tf.nn.relu(conv2d_1(h_norm2, mnist_weights['wc3'] ) + mnist_biases['bc3'])
		h_norm3 = norm(h_conv3)

		#print('\n*******h_conv3 is: ' + str(h_conv3))
		#print('\n*******h_conv3 type is: ' + str(type(h_conv3)))
		h_conv4 = tf.nn.relu(conv2d_1(h_norm3, mnist_weights['wc4'] ) + mnist_biases['bc4'])
		h_norm4 = norm(h_conv4)

		#print('\n*******h_conv4 is: ' + str(h_conv4))
		#print('\n*******h_conv4 type is: ' + str(type(h_conv4)))

		#following is old:
		h_conv5 = tf.nn.relu(conv2d_1(h_conv4, mnist_weights['wc5'] ) + mnist_biases['bc5'])
		#h_conv5 = tf.nn.relu(conv2d_p(h_norm4, weights['wc5'] , 2) + biases['bc5'])

		h_pool5 = max_pool_3x3(h_conv5)
		h_norm5 = norm(h_pool5)

		#print('\n*******h_conv5 is: ' + str(h_conv5))
		#print('\n*******h_conv5 type is: ' + str(type(h_conv5)))

		#print('\n*******h_pool5 is: ' + str(h_pool5))
		#print('\n*******h_pool5 type is: ' + str(type(h_pool5)))


		#layer 4
		h_conv6 = tf.reshape(h_norm5, [-1, mnist_weights['wd1'].get_shape().as_list()[0]])


		#print('\n*******h_conv6 is: ' + str(h_conv6))
		#print('\n*******h_conv6 type is: ' + str(type(h_conv6)))


		h_fc1 = tf.nn.relu(tf.matmul(h_conv6, mnist_weights['wd1']) + mnist_biases['bd1'])
		h_fc1 = dropout(h_fc1)


		#print('\n*******h_fc1 is: ' + str(h_fc1))
		#print('\n*******h_fc1 type is: ' + str(type(h_fc1)))


		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, mnist_weights['wd2']) + mnist_biases['bd2'])
		h_fc2 = dropout(h_fc2)
		#print('\n*******h_fc2 is: ' + str(h_fc2))
		#print('\n*******h_fc2 type is: ' + str(type(h_fc2)))


		h_fc3 = tf.matmul(h_fc2 , mnist_weights['out']) + mnist_biases['out']
		#print('\n*******h_fc3 is: ' + str(h_fc3))
		#print('\n*******h_fc3 type is: ' + str(type(h_fc3)))


		loss   = loss_cross_entropy(h_fc3 , mnist_y)




		optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(mnist_y , 1))
		accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))


		saver = tf.train.Saver()
		#print('\n+++++++++++ saver is: ' + str(saver) + '\n')

		return(h_fc3)


def working_flow_cifar10():
		global accuracy
		global saver
		global optimizer_1
		global loss

		#layer 0
		x_image = tf.reshape(cifar10_x, [-1,32,32,3])


		#layer 1
		h_conv1 = tf.nn.relu(conv2d_1(x_image, cifar10_weights['wc1']) + cifar10_biases['bc1'])


		print('\n*******h_conv1 is: ' + str(h_conv1))
		print('\n*******h_conv1 type is: ' + str(type(h_conv1)))
		h_pool1 = max_pool_3x3(h_conv1)


		print('\n*******h_pool1 is: ' + str(h_pool1))
		print('\n*******h_pool1 type is: ' + str(type(h_pool1)))


		h_norm1 = norm(h_pool1)
		h_norm1 = h_pool1


		#layer 2
		h_conv2 = tf.nn.relu(conv2d_1(h_norm1, cifar10_weights['wc2'] ) + cifar10_biases['bc2'])

		print('\n*******h_conv2 is: ' + str(h_conv2))
		print('\n*******h_conv2 type is: ' + str(type(h_conv2)))
		h_pool2 = max_pool_3x3(h_conv2)

		print('\n*******h_pool2 is: ' + str(h_pool2))
		print('\n*******h_pool2 type is: ' + str(type(h_pool2)))

		h_norm2 = norm(h_pool2)
		h_norm2 = h_pool2

		############################
		#layer 3
		h_conv3 = tf.nn.relu(conv2d_1(h_norm2, cifar10_weights['wc3'] ) + cifar10_biases['bc3'])
		h_norm3 = norm(h_conv3)
		h_norm3 = h_conv3



		print('\n*******h_conv3 is: ' + str(h_conv3))
		print('\n*******h_conv3 type is: ' + str(type(h_conv3)))
		h_conv4 = tf.nn.relu(conv2d_1(h_norm3, cifar10_weights['wc4'] ) + cifar10_biases['bc4'])
		h_norm4 = norm(h_conv4)
		h_norm4 = h_conv4




		print('\n*******h_conv4 is: ' + str(h_conv4))
		print('\n*******h_conv4 type is: ' + str(type(h_conv4)))

		#following is old:
		h_conv5 = tf.nn.relu(conv2d_1(h_conv4, cifar10_weights['wc5'] ) + cifar10_biases['bc5'])
		#h_conv5 = tf.nn.relu(conv2d_p(h_norm4, weights['wc5'] , 2) + biases['bc5'])

		h_pool5 = max_pool_3x3(h_conv5)
		h_norm5 = norm(h_pool5)
		h_norm5 = h_pool5




		print('\n*******h_conv5 is: ' + str(h_conv5))
		print('\n*******h_conv5 type is: ' + str(type(h_conv5)))

		print('\n*******h_pool5 is: ' + str(h_pool5))
		print('\n*******h_pool5 type is: ' + str(type(h_pool5)))


		#layer 4
		h_conv6 = tf.reshape(h_norm5, [-1, cifar10_weights['wd1'].get_shape().as_list()[0]])


		print('\n*******h_conv6 is: ' + str(h_conv6))
		print('\n*******h_conv6 type is: ' + str(type(h_conv6)))


		h_fc1 = tf.nn.relu(tf.matmul(h_conv6, cifar10_weights['wd1']) + cifar10_biases['bd1'])
		h_fc1 = dropout(h_fc1)


		print('\n*******h_fc1 is: ' + str(h_fc1))
		print('\n*******h_fc1 type is: ' + str(type(h_fc1)))


		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, cifar10_weights['wd2']) + cifar10_biases['bd2'])
		h_fc2 = dropout(h_fc2)
		print('\n*******h_fc2 is: ' + str(h_fc2))
		print('\n*******h_fc2 type is: ' + str(type(h_fc2)))


		h_fc3 = tf.matmul(h_fc2 , cifar10_weights['out']) + cifar10_biases['out']
		print('\n*******h_fc3 is: ' + str(h_fc3))
		print('\n*******h_fc3 type is: ' + str(type(h_fc3)))


		loss   = loss_cross_entropy(h_fc3 , cifar10_y)
		#print('h_fc3 is:' + str(h_fc3.get_shape()))


		#最优控制
		#优化器---->各种对于梯度下降算法的优化。
		optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(cifar10_y , 1))
		accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))
		#print('accuracy is:' + str(accuracy.get_shape()))

		saver = tf.train.Saver()
		#print('\n+++++++++++ saver is: ' + str(saver) + '\n')

		return(h_fc3)


#-----------training start -----------#


def cnn_train_mnist():
		with tf.Session() as sess:
				train_acc = []
				sess = tf.InteractiveSession()
				sess.run(tf.global_variables_initializer())
				for i in range(100):
						#begin_1 = time.time()


						batch = mnist.train.next_batch(64)
						batch_1 = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')
						#type of batch and batch_1 is tuple
						#batch_1[0][0] 和 batch[0] 都是图片

						#print(batch_1[0][0])

						#print( 'batch type is: ' + str(type(batch)))
						#print( 'batch_1 type is: ' + str(type(batch_1)))



						#plt.imshow(np.reshape((batch_1[0][i]/256), [32 , 32 , 3]) )

						#plt.show()

						'''

						plt.imshow(batch[0])
						plt.show()

						#print( 'batch is: ' + str(batch))
						#print( 'batch_1 is: ' + str(batch_1))

						#print( 'batch type is: ' + str(type(batch)))
						#print( 'batch_1 type is: ' + str(type(batch_1)))

						print('\n^^^^^^^^batch[0] is :' + str(batch[0]) + '\n')
						print('\n^^^^^^^^batch[1] is :' + str(batch[1]) + '\n')
						print('\n^^^^^^^^batch[0] length is :' + str(len(batch[0])) + '\n')
						print('\n^^^^^^^^batch[1] length is :' + str(len(batch[1])) + '\n')

						print('\n^^^^^^^^accuracy is :' + str(accuracy) + '\n')
						#print('\n^^^^^^^^accuracy length is :' + str(len(accuracy)) + '\n')
						'''


						if i%5 == 0:
								#begin_2 = time.time()

								#train_accuracy = accuracy.eval(feed_dict={mnist_x:batch[0] , mnist_y:batch[1], keep_prob:1.0})
								train_accuracy = sess.run(accuracy, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})
								#train_accuracy = sess.run(accuracy, feed_dict={mnist_x: batch_1[0][0], mnist_y: batch_1[1][0], keep_prob: 1.})

								loss_1 = sess.run(loss, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})


								train_acc.append(train_accuracy)
								#print ("step %d, training accuracy %g" % (i, train_accuracy))

								# save model for reusing
								saver.save(sess, './../model/model_alexnet_mnist.ckpt')
								#end_2 = time.time()
								#print('Using time 2 :' + str(end_2 - begin_2) + '\n')


								#print ("Iter " + str((i+1)*64) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy))
								print ("Iter " + str((i+1)*64) + ", Minibatch Loss = " + str(loss_1) + ", Training Accuracy = " + str(train_accuracy))


								#plt.title('Actual: ' + label_2_figure(str(batch[1][0])) + '        Predict: ' + label_2_figure(str(batch[1][i])) + '\n' + "Iter " + str((i+1)*64) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy))
								#plt.title('Actual: ' + label_2_figure(str(batch[1][0])) + '        Predict: ' + label_2_figure(str(batch[1][i])) )
								#plt.imshow(np.reshape(batch[0][0] , [28 , 28]) )
								#plt.show()

						# 优化步骤
						optimizer_1.run(feed_dict = {mnist_x:batch[0] , mnist_y:batch[1] , keep_prob:0.5})


						#end_1 = time.time()
						#print('Using time 1 :' + str(end_1 - begin_1) + '\n')
				#print('optimizer is: ' + str(optimizer))
				#plt.plot(indice, train_acc, 'k-', label = "Train Set Accuracy")
				#plt.title('Train Accuracy')
				#plt.xlabel('Generation')
				#plt.ylabel('Accuracy')
				#plt.show()


def cnn_train_cifar10():
		global learning_rate
		with tf.Session() as sess:
				train_acc = []
				sess = tf.InteractiveSession()
				sess.run(tf.global_variables_initializer())
				for i in range(10000):
						#begin_1 = time.time()


						batch_mnist    = mnist.train.next_batch(batch_size)
						batch_1 = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')


						#batch_cifar10  = cifar10.train.next_batch(64)
						'''
						#调试新建batch函数的log , 查看生成的batch是否正确

						print('\n----------------------------------------cifar10[1]        is:' +  str(cifar10[1]))
						print('\n----------------------------------------cifar10[1] length is:' +  str(len(cifar10[1])))

						print('\n----------------------------------------cifar10[1][1]     is:' +  str(cifar10[1][1]))


						#type of batch and batch_1 is tuple
						#batch_1[0][0] 和 batch[0] 都是图片
						'''
						batch_cifar10  = batch_next(cifar10[0], cifar10[1] , batch_size)


						#print( 'batch type is: ' + str(type(batch)))


						# batch_1[1]    是 10000个元素的向量，存储了10000个类别的值
						# batch_1[0]    是 10000个元素的向量，存储了10000个矩阵，矩阵是32x32x3

						# batch_1[0][0] 是32x32x3 的矩阵
						# batch_1[1][0] 是分类的数值

						#print('batch_cifar10    is: ' + str(batch_cifar10))

						'''
						print('batch_cifar10[0] is: ' + str(batch_cifar10[0]))

						print('batch_cifar10[0] length is: ' + str(len(batch_cifar10[0])))
						print('batch_mnist[0]   length is: ' + str(len(batch_mnist[0])))
						print('batch_cifar10[0] type is: ' + str(type(batch_cifar10[0])))
						print('batch_mnist[0]   type is: ' + str(type(batch_mnist[0])))
						print('batch_mnist[0][0]length is: ' + str(len(batch_mnist[0][0])))


						print('batch_cifar10[0][0] length is: '    + str(len(batch_cifar10[0][0])))

						#print('batch_cifar10[1][0]  is: '    + str(batch_cifar10[1][0]))
						print('batch_cifar10[1]  is: '    + str(batch_cifar10[1]))





						#print('batch_cifar10[0][0][0] length is: ' + str(len(batch_cifar10[0][0][0])))





						print('batch_cifar10 is:' + str(batch_cifar10))
						print('batch_mnist is:' + str(batch_mnist))

						#plt.imshow(np.reshape((batch_1[0][i]/256), [32 , 32 , 3]) )
						'''

						'''
						#图片显示调试
						plt.imshow(np.reshape((batch_cifar10[0][i]/256) , [32 , 32 , 3]))
						plt.show()
						'''

						'''
						#调试新建batch函数的log , 查看生成的batch是否正确

						print('batch_mnist[0]   is: ' + str(batch_mnist[0]))
						print('batch_cifar10[0] is: ' + str(batch_cifar10[0]))



						print('batch_mnist[1]   is: ' + str(batch_mnist[1]))
						print('batch_mnist[1][0]   is: ' + str(batch_mnist[1][0]))
						print('batch_mnist[1][0] type is: ' + str(type(batch_mnist[1][0])))


						print('batch_cifar10[1] is: ' + str(batch_cifar10[1]))
						#print('batch_cifar10[1][0] is: ' + str(batch_cifar10[1][0]))

						print('batch_cifar10[1] type is: ' + str(type(batch_cifar10[1])))
						#print('batch_cifar10[1][0] type is: ' + str(type(batch_cifar10[1][0])))

						print('batch_cifar10[1][1] type is: ' + str(type(batch_cifar10[1][1])))
						'''



						# 优化器步骤

						#optimizer_1.run(feed_dict = {cifar10_x: batch_cifar10[0] , cifar10_y: batch_cifar10[1] , keep_prob:0.75})

						if i%5 == 0:
								#begin_2 = time.time()
								'''
								print('cifar10_x is: '     + str(cifar10_x))
								print('batch_cifar10 is: ' + str(batch_cifar10))

								print('mnist_x is: '     + str(mnist_x))
								print('batch_mnist is: ' + str(batch_mnist))
								'''

								train_accuracy = sess.run(accuracy, feed_dict={cifar10_x: batch_cifar10[0], cifar10_y: batch_cifar10[1], keep_prob: 1.})
								#train_accuracy = sess.run(accuracy, feed_dict={cifar10_x: batch_1[0], cifar10_y: batch_1[1], keep_prob: 1.})

								loss_1         = sess.run(loss,     feed_dict={cifar10_x: batch_cifar10[0], cifar10_y: batch_cifar10[1], keep_prob: 1.})


								train_acc.append(train_accuracy)
								#print ("step %d, training accuracy %g" % (i, train_accuracy))

								# save model for reusing
								saver.save(sess, './../model/model_alexnet_cifar10.ckpt')
								#end_2 = time.time()
								#print('Using time 2 :' + str(end_2 - begin_2) + '\n')


								print ("Iter " + str((i+1)*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy) + ', Learning_rate is:  ' + str(learning_rate))


								#plt.title('Actual: ' + label_2_figure(str(batch[1][0])) + '        Predict: ' + label_2_figure(str(batch[1][i])) + '\n' + "Iter " + str((i+1)*64) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy))
								#plt.title('Actual: ' + label_2_figure(str(batch[1][0])) + '        Predict: ' + label_2_figure(str(batch[1][i])) )
								#plt.imshow(np.reshape(batch[0][0] , [28 , 28]) )
								#plt.show()

								if i<50:
									learning_rate = learning_rate_init
								elif i>50  and i<100:
									learning_rate = learning_rate_init / 10
								elif i>100 and i<200:
									learning_rate = learning_rate_init / 100
								elif i>200 and i<400:
									learning_rate = learning_rate_init / 1000
								elif i>400 and i<600:
									learning_rate = learning_rate_init / 10000
								else:
									learning_rate = learning_rate_init / 100000



						optimizer_1.run(feed_dict = {cifar10_x: batch_cifar10[0] , cifar10_y: batch_cifar10[1] , keep_prob:1.})

						#end_1 = time.time()
						#print('Using time 1 :' + str(end_1 - begin_1) + '\n')
				#print('optimizer is: ' + str(optimizer))
				#plt.plot(indice, train_acc, 'k-', label = "Train Set Accuracy")
				#plt.title('Train Accuracy')
				#plt.xlabel('Generation')
				#plt.ylabel('Accuracy')
				#plt.show()


'''
def cnn_train():
		with tf.Session() as sess:
				#sess.run(tf.global_variables_initializer())
				sess.run(tf.initialize_all_variables())
				# Keep training until reach max iterations
				#while step * batch_size < training_iters:
				for i in range(5000):
						batch_xs, batch_ys = mnist.train.next_batch(64)

						sess.run(optimizer_1, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 0.75})

						if i % 5 == 0:
								acc = sess.run(accuracy, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 1.})
								loss_1 = sess.run(loss, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 1.})

								print('loss_1 is :' + str(loss_1))
								print('loss_1 type is :' + str(type(loss_1)))

								print('loss is :' + str(loss))
								print('loss type is :' + str(type(loss)))


								print ("Iter " + str(i) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(acc))

				print ("Optimization Finished!")
				print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
'''


#predict
def predict():
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess, './../model/model_alexnet_cifar10.ckpt')
		print( "test accuracy %g" % accuracy.eval(feed_dict={mnist_x:mnist.test.images, mnist_y:mnist.test.labels, keep_prob:1.0}))


#-----------Main program start -----------#

if __name__ == '__main__':
		#working_flow_mnist()
		working_flow_cifar10()

		saver = tf.train.Saver()
		#print('\n+++++++++++ saver is: ' + str(saver) + '\n')

		#cnn_train_mnist()
		cnn_train_cifar10()
		predict()
