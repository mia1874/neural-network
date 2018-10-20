# -*- coding = utf8 -*-

'''#------------------------network info---------------------------------#
20181008
AlexNet


Type:            Maps   Size       Kernelsize   Stride  Padding

input            3      32x32      -            -       -

conv3-16		
max_pool		

conv3-32		
max_pool		

conv3-64		
max_pool		

conv3-128		
conv3-128		

conv3-256		
conv3-256		

reshape			
fully connected	
dropout			

fully connected	
dropout			

fully connected	
softmax out		
#---------------------------------------------------------------------#'''


file_info = (
		'''
		VGG
		Using for MNIST , CIFAR-10
		Author: MrZQ
		20181019


		v1.0.0.1019_alpha Update
				1. VGG working on MNIST and CIFAR10 data set
				2. Include 4 files
				3. Bug fix
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
import cifar10_load



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





# hyperparameter for cifar10 using VGG-16
cifar10_weights = {
		'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 16])),
		'wc2': tf.Variable(tf.truncated_normal([3, 3, 16, 32])),
		'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 64])),
		'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
		'wc5': tf.Variable(tf.truncated_normal([3, 3, 64, 128])),
		'wc6': tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
		'wc7': tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
		'wc8': tf.Variable(tf.truncated_normal([3, 3, 128, 128])),



		#'wd1': tf.Variable(tf.truncated_normal([n_in, 256])),
		# n_in is 1152
		

		'wd1': tf.Variable(tf.truncated_normal([1152, 256])),
		'wd2': tf.Variable(tf.truncated_normal([256, 256])),
		'out': tf.Variable(tf.truncated_normal([256, 10]))
}
cifar10_biases = {
		'bc1': tf.Variable(tf.truncated_normal([16])),
		'bc2': tf.Variable(tf.truncated_normal([32])),
		'bc3': tf.Variable(tf.truncated_normal([64])),
		'bc4': tf.Variable(tf.truncated_normal([64])),
		'bc5': tf.Variable(tf.truncated_normal([128])),
		'bc6': tf.Variable(tf.truncated_normal([128])),
		'bc7': tf.Variable(tf.truncated_normal([128])),
		'bc8': tf.Variable(tf.truncated_normal([128])),
		
		'bd1': tf.Variable(tf.truncated_normal([256])),
		'bd2': tf.Variable(tf.truncated_normal([256])),
		'out': tf.Variable(tf.truncated_normal([10]))
}




'''
mnist       = input_data.read_data_sets('./../../data/MNIST_data' , one_hot = True)
mnist_x     = tf.placeholder("float", shape = [None, 784])
mnist_y     = tf.placeholder("float", shape = [None, 10])
'''



cifar10      = cifar10_load.load_CIFAR10('./../../data/cifar-10-batches-py')
cifar10_test = cifar10_load.load_CIFAR_batch('./../../data/cifar-10-batches-py/test_batch')

'''
old parameter
#cifar10_x   = tf.placeholder("float", shape = [None, 32 ,32 ,3])
cifar10_x    = tf.placeholder("float", shape = [None, 3072])
cifar10_y    = tf.placeholder("float", shape = [None, 10])
'''



cifar10_x    = tf.placeholder(tf.float32, shape = [None, 24 ,24 ,3])
cifar10_y    = tf.placeholder(tf.float32, shape = [None, 256])



#-------------function init-----------------#

def weight_variable(shape):
		#initial = tf.truncated_normal(shape , stddev=0.1)
		initial = tf.truncated_normal(shape , stddev=0.01)
		return tf.Variable(initial)

'''
new 
'''




def bias_variable(shape):
		#initial = tf.constant(0.1 , shape = shape)
		initial = tf.constant(0.0 , shape = shape)
		return tf.Variable(initial)

def conv2d_p(x ,W ,s):
		return tf.layers.conv2d(x, W, strides=[1,s,s,1] , padding = 'SAME')

def conv2d_1(x ,W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1] , padding = 'SAME')

def max_pool_3x3(x):
		return tf.nn.max_pool(x,  ksize=[1,2,2,1] , strides=[1,2,2,1] , padding = "SAME")

#def max_pool_3x3(x):
#		return tf.nn.max_pool(x,  ksize=[1,2,2,1] , strides=[1,2,2,1] , padding = "SAME")


def norm(x):
		return tf.nn.lrn(x , 4 , bias = 1.0 , alpha = 0.001/9.0 , beta = 0.75)


def dropout(x):
		return(tf.nn.dropout(x , keep_prob))

def softmax(x, W, b):
		return(tf.nn.softmax(tf.matmul(x, W) + b))

def loss_cross_entropy( x,  y):
		#===============================change loss function
		return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x , labels = y)))
		
		
		
		#return (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = x , labels = y)))
		#return (-tf.reduce_mean(y * tf.log(tf.clip_by_value(x, 1e-10, 1.0))) 

def optimizer(learning_rate, cross_entropy):
		return(tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy))

def correct_predict(a,b):
		return(tf.equal(tf.argmax(a,1) , tf.argmax(b,1)))

def accuracy(predict):
		return(tf.reduce_mean(tf.cast(predict, "float")))


#---------------- working flow----------------#
def working_flow_cifar10():
		global accuracy
		global saver
		global optimizer_1
		global loss
		global predict
		global n_in

		#layer 0
		x_image = tf.reshape(cifar10_x, [-1,24,24,3])


		#============layer 1=================#
		#============conv + pool=============#
		#h_conv1 = tf.nn.relu(conv2d_1(x_image, cifar10_weights['wc1']) + cifar10_biases['bc1'])
		h_conv1  = conv2d_1(x_image, cifar10_weights['wc1']) + cifar10_biases['bc1']
		bn_conv1 = tf.layers.batch_normalization(h_conv1, training=is_training, name='bn1')
		bn_conv1 = tf.nn.relu(bn_conv1)


		print('\n*******h_conv1 is: ' + str(h_conv1))
		print('\n*******h_conv1 type is: ' + str(type(h_conv1)))
		#=================================pooling 1 
		h_pool1 = max_pool_3x3(bn_conv1)


		print('\n*******h_pool1 is: ' + str(h_pool1))
		print('\n*******h_pool1 type is: ' + str(type(h_pool1)))


		h_norm1 = norm(h_pool1)
		#h_norm1 = h_pool1


		#============layer 2=================#
		#============conv + pool=============#
		#h_conv2 = tf.nn.relu(conv2d_1(h_norm1, cifar10_weights['wc2'] ) + cifar10_biases['bc2'])
		h_conv2  = conv2d_1(h_norm1, cifar10_weights['wc2']) + cifar10_biases['bc2']
		bn_conv2 = tf.layers.batch_normalization(h_conv2, training=is_training, name='bn2')
		bn_conv2 = tf.nn.relu(bn_conv2)




		print('\n*******h_conv2 is: ' + str(h_conv2))
		print('\n*******h_conv2 type is: ' + str(type(h_conv2)))
		#=================================pooling 2 
		h_pool2 = max_pool_3x3(bn_conv2)

		print('\n*******h_pool2 is: ' + str(h_pool2))
		print('\n*******h_pool2 type is: ' + str(type(h_pool2)))

		h_norm2 = norm(h_pool2)
		#h_norm2 = h_pool2



		#============layer 3=================#
		#============conv + conv + pool======#
		#h_conv3 = tf.nn.relu(conv2d_1(h_norm2, cifar10_weights['wc3'] ) + cifar10_biases['bc3'])
		h_conv3  = conv2d_1(h_norm2, cifar10_weights['wc3']) + cifar10_biases['bc3']
		bn_conv3 = tf.layers.batch_normalization(h_conv3, training=is_training, name='bn3')
		bn_conv3 = tf.nn.relu(bn_conv3)

		h_norm3  = norm(bn_conv3)
		#h_norm3 = h_conv3



		print('\n*******h_conv3 is: ' + str(h_conv3))
		print('\n*******h_conv3 type is: ' + str(type(h_conv3)))
		#h_conv4 = tf.nn.relu(conv2d_1(h_norm3, cifar10_weights['wc4'] ) + cifar10_biases['bc4'])
		h_conv4  = conv2d_1(h_norm3, cifar10_weights['wc4']) + cifar10_biases['bc4']
		bn_conv4 = tf.layers.batch_normalization(h_conv4, training=is_training, name='bn4')
		bn_conv4 = tf.nn.relu(bn_conv4)


		#=================================pooling 3 
		h_pool3 = max_pool_3x3(bn_conv4)

		
		
		h_norm4 = norm(h_pool3)
		#h_norm4 = h_conv4
		print('\n*******h_conv4 is: ' + str(h_conv4))
		print('\n*******h_conv4 type is: ' + str(type(h_conv4)))

		#following is old:
		#h_conv5 = tf.nn.relu(conv2d_1(h_conv4, cifar10_weights['wc5'] ) + cifar10_biases['bc5'])
		#h_conv5 = tf.nn.relu(conv2d_p(h_norm4, weights['wc5'] , 2) + biases['bc5'])



		#============layer 4=================#
		#============conv + conv ============#


		h_conv5  = conv2d_1(h_norm4, cifar10_weights['wc5']) + cifar10_biases['bc5']
		bn_conv5 = tf.layers.batch_normalization(h_conv5, training=is_training, name='bn5')
		bn_conv5 = tf.nn.relu(bn_conv5)



		#h_pool5 = max_pool_3x3(bn_conv5)
		#h_norm5 = norm(h_pool5)
		h_norm5 = norm(bn_conv5)

		
		#h_norm5 = h_pool5


		print('\n*******h_conv5 is: ' + str(h_conv5))
		print('\n*******h_conv5 type is: ' + str(type(h_conv5)))

		print('\n*******h_norm5 is: ' + str(h_norm5))
		print('\n*******h_norm5 type is: ' + str(type(h_norm5)))




		h_conv6  = conv2d_1(h_norm5, cifar10_weights['wc6']) + cifar10_biases['bc6']
		bn_conv6 = tf.layers.batch_normalization(h_conv6, training=is_training, name='bn6')
		bn_conv6 = tf.nn.relu(bn_conv6)



		#h_pool6 = max_pool_3x3(bn_conv6)
		#h_norm6 = norm(h_pool6)
		h_norm6 = norm(bn_conv6)
		
		
		#h_norm5 = h_pool5


		print('\n*******h_conv6 is: ' + str(h_conv6))
		print('\n*******h_conv6 type is: ' + str(type(h_conv6)))

		print('\n*******h_norm6 is: ' + str(h_norm6))
		print('\n*******h_norm6 type is: ' + str(type(h_norm6)))

	
		
		#============layer 5=================#
		#============conv + conv ============#
		h_conv7  = conv2d_1(h_norm6, cifar10_weights['wc7']) + cifar10_biases['bc7']
		bn_conv7 = tf.layers.batch_normalization(h_conv7, training=is_training, name='bn7')
		bn_conv7 = tf.nn.relu(bn_conv7)

		#h_pool7 = max_pool_3x3(bn_conv7)
		#h_norm7 = norm(h_pool7)
		h_norm7 = norm(bn_conv7)
		
		#h_norm5 = h_pool5


		print('\n*******h_conv7 is: ' + str(h_conv7))
		print('\n*******h_conv7 type is: ' + str(type(h_conv7)))

		print('\n*******h_norm7 is: ' + str(h_norm7))
		print('\n*******h_norm7 type is: ' + str(type(h_norm7)))


		h_conv8  = conv2d_1(h_norm7, cifar10_weights['wc8']) + cifar10_biases['bc8']
		bn_conv8 = tf.layers.batch_normalization(h_conv8, training=is_training, name='bn8')
		bn_conv8 = tf.nn.relu(bn_conv8)

		#h_pool8 = max_pool_3x3(bn_conv8)
		#h_norm8 = norm(h_pool8)
		h_norm8 = norm(bn_conv8)
		#h_norm5 = h_pool5


		print('\n*******h_conv8 is: ' + str(h_conv8))
		print('\n*******h_conv8 type is: ' + str(type(h_conv8)))

		print('\n*******h_norm8 is: ' + str(h_norm8))
		print('\n*******h_norm8 type is: ' + str(type(h_norm8)))




		#reshape_norm8 = tf.reshape(h_norm8, [batch_size, -1])
		reshape_norm8 = tf.reshape(h_norm8, [256, -1])

		n_in = reshape_norm8.get_shape()[-1].value

		print('n_in is: ' + str(n_in) + '\n')
		

	
		#conv8 and fully_connect
		#-----------------------------------------------------------------------------------------------------------------------------------------------------
		'''
		#layer 4
		h_conv6 = tf.reshape(h_norm5, [-1, cifar10_weights['wd1'].get_shape().as_list()[0]])


		print('\n*******h_conv6 is: ' + str(h_conv6))
		print('\n*******h_conv6 type is: ' + str(type(h_conv6)))


		#h_fc1 = tf.nn.relu(tf.matmul(h_conv6, cifar10_weights['wd1']) + cifar10_biases['bd1'])
		#======================= remove dropout begin
		'''
		#-----------------------------------------------------------------------------------------------------------------------------------------------------


		h_fc1  = tf.matmul(reshape_norm8 , cifar10_weights['wd1']) + cifar10_biases['bd1']
		bn_fc1 = tf.layers.batch_normalization(h_fc1, training=is_training, name='fc1')
		bn_fc1 = tf.nn.relu(bn_fc1)
		#h_fc1 = dropout(bn_fc1)


		print('\n*******h_fc1 is: ' + str(h_fc1))
		print('\n*******h_fc1 type is: ' + str(type(h_fc1)))


		#h_fc2 = tf.nn.relu(tf.matmul(h_fc1, cifar10_weights['wd2']) + cifar10_biases['bd2'])
		#======================= remove dropout begin
		#h_fc2 = dropout(h_fc2)
		h_fc2  = tf.matmul(bn_fc1, cifar10_weights['wd2']) + cifar10_biases['bd2']
		bn_fc2 = tf.layers.batch_normalization(h_fc2, training=is_training, name='fc2')
		bn_fc2 = tf.nn.relu(bn_fc2)

		print('\n*******h_fc2 is: ' + str(h_fc2))
		print('\n*******h_fc2 type is: ' + str(type(h_fc2)))


		h_fc3 = tf.matmul(bn_fc2 , cifar10_weights['out']) + cifar10_biases['out']
		print('\n*******h_fc3 is: ' + str(h_fc3))


		loss  = loss_cross_entropy( h_fc3 , cifar10_y)
		#loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = h_fc3 , labels = cifar10_y))


		#最优控制
		#优化器---->各种对于梯度下降算法的优化。
		#似乎没什么区别
		optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		
		#optimizer_1     = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)
		
		
		correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(cifar10_y , 1))
		#accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))
		accuracy        = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

		predict         = tf.argmax(h_fc3 , 1)


		#print('accuracy is:' + str(accuracy.get_shape()))

		saver = tf.train.Saver()
		#print('\n+++++++++++ saver is: ' + str(saver) + '\n')
		return(h_fc3)





#---------------- training start -------------#
def cnn_train_cifar10():
		global learning_rate
		'''
		with tf.Session(config=tf.ConfigProto(
						device_count={"CPU":3},
						inter_op_parallelism_threads=1,
						intra_op_parallelism_threads=1,
						)) as sess:
		'''	
		with tf.Session() as sess:
				train_acc = []
				sess = tf.InteractiveSession()
				sess.run(tf.global_variables_initializer())
				for i in range(1000000):
						#begin_1 = time.time()


						#batch_mnist    = mnist.train.next_batch(batch_size)
						#batch_1 = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')

						#batch_cifar10  = cifar10.train.next_batch(64)
						'''
						#调试新建batch函数的log , 查看生成的batch是否正确

						print('\n----------------------------------------cifar10[1]        is:' +  str(cifar10[1]))
						print('\n----------------------------------------cifar10[1] length is:' +  str(len(cifar10[1])))

						print('\n----------------------------------------cifar10[1][1]     is:' +  str(cifar10[1][1]))


						#type of batch and batch_1 is tuple
						#batch_1[0][0] 和 batch[0] 都是图片
						'''
						batch_cifar10  = cifar10_load.batch_next(cifar10[0], cifar10[1] , batch_size)
						
						#图片及label生成测试
						'''
						*****************************************************
						plt.title('Iter: '+ str(i) + '  Actual: ' + label_2_figure(str(batch_cifar10[1][i])))
						plt.imshow(np.reshape((batch_cifar10[0][i]/256) , [32 , 32 , 3]))
						plt.show()
						*****************************************************
						'''

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
						

						
						#图片显示调试
						plt.title('Actual: ' + label_2_figure(str(batch[1][0])) + '     Predict: ' + label_2_figure(str(batch_cifar10[1][i])) + '\n')

						plt.imshow(np.reshape((batch_cifar10[0][i]/256) , [32 , 32 , 3]))
						plt.show()
						

						
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
								
								print('cifar10_x is: '     + str(cifar10_x))
								print('batch_cifar10 is: ' + str(batch_cifar10))
								'''
								print('mnist_x is: '     + str(mnist_x))
								print('batch_mnist is: ' + str(batch_mnist))
								'''

								train_accuracy = sess.run(accuracy, feed_dict={cifar10_x: batch_cifar10[0], cifar10_y: batch_cifar10[1], keep_prob: 1.})
								#train_accuracy = sess.run(accuracy, feed_dict={cifar10_x: batch_1[0], cifar10_y: batch_1[1], keep_prob: 1.})
								#print('train_accuracy is: '      + str(train_accuracy))
								#print('train_accuracy type is: ' + str(type(train_accuracy)))


								loss_1         = sess.run(loss,     feed_dict={cifar10_x: batch_cifar10[0], cifar10_y: batch_cifar10[1], keep_prob: 1.})
								prediction     = sess.run(predict,  feed_dict={cifar10_x: batch_cifar10[0], cifar10_y: batch_cifar10[1], keep_prob: 1.})
								#print('=-=================== loss :' + str(loss_1) + '\n')


								train_acc.append(train_accuracy)
								#print ("step %d, training accuracy %g" % (i, train_accuracy))

								# save model for reusing
								saver.save(sess, './../model/model_alexnet_cifar10.ckpt')
								#end_2 = time.time()
								#print('Using time 2 :' + str(end_2 - begin_2) + '\n')



								#print ("Iter " + str((i+1)*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy) + ', Learning_rate is:  ' + str(learning_rate))
								print ("Iter " + str((i+1)*batch_size) + ", Minibatch Loss = " + str(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy) + ', Learning_rate is:  ' + str(learning_rate))

								'''
								plt.title('Actual: ' + label_2_figure(str(batch_cifar10[1][i])) + '     Predict: ' + str(prediction[i]) + '\n')
								plt.imshow(np.reshape((batch_cifar10[0][i]/256) , [32 , 32 , 3]))
								plt.show()
								'''


								if i<500:
									learning_rate = learning_rate_init
								elif i>500  and i<1000:
									learning_rate = learning_rate_init / 10
								elif i>1000 and i<2000:
									learning_rate = learning_rate_init / 100
								elif i>2000 and i<5000:
									learning_rate = learning_rate_init / 1000
								elif i>5000 and i<10000:
									learning_rate = learning_rate_init / 10000
								else:
									learning_rate = learning_rate_init / 100000

								if i == 10000:
									sess.close()
									break


						optimizer_1.run(feed_dict = {cifar10_x: batch_cifar10[0] , cifar10_y: batch_cifar10[1] , keep_prob:1.})

						#end_1 = time.time()
						#print('Using time 1 :' + str(end_1 - begin_1) + '\n')
				#print('optimizer is: ' + str(optimizer))
				#plt.plot(indice, train_acc, 'k-', label = "Train Set Accuracy")
				#plt.title('Train Accuracy')
				#plt.xlabel('Generation')
				#plt.ylabel('Accuracy')
				#plt.show()



#----------------- predict start -------------#
def predict_cifar10():
		batch_size = 1000
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess, './../model/model_alexnet_cifar10.ckpt')


		#mnist 的test batch
		test_batch    = batch_next(cifar10_test[0], cifar10_test[1] , batch_size)

		
		loss_3        = sess.run(loss,     feed_dict={cifar10_x: test_batch[0], cifar10_y: test_batch[1], keep_prob: 1.})
		test_accuracy = sess.run(accuracy, feed_dict={cifar10_x: test_batch[0], cifar10_y: test_batch[1], keep_prob: 1.})
		

		#for i in range(5):
		print ("Iter " + str(batch_size) + ", Minibatch Loss = " + str(loss_3) + ", Testing Accuracy = " + str(test_accuracy) )
		
		#print( "test accuracy %g" % accuracy.eval(feed_dict={mnist_x:mnist.test.images, mnist_y:mnist.test.labels, keep_prob:1.0}))




#-----------Main program start -----------#

if __name__ == '__main__':
		working_flow_cifar10()

		saver = tf.train.Saver()

		cnn_train_cifar10()
		
		predict_cifar10()



