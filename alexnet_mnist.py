#-*- coding : utf-8 -*-

'''#-------------------------------------------------------------#
20181003
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
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time


#-------------variable init-----------------#
mnist = input_data.read_data_sets('./../data/MNIST_data' , one_hot = True)
mnist_x  = tf.placeholder("float", shape = [None, 784])
mnist_y  = tf.placeholder("float", shape = [None, 10])
keep_prob = tf.placeholder("float")
#keep_prob = 0.75
#learning_rate = 0.0001-----> 0.1
#learning_rate = 0.01  -----> 0.2 0.03
learning_rate = 0.001
learning_rate_init = 0.001
is_training = 1
batch_size = 100


accuracy = []
saver = []
optimizer_1 = []
loss = []



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

#-------------function init-----------------#
def weight_variable(shape):
		initial = tf.truncated_normal(shape , stddev=0.1)
		return tf.Variable(initial)

def bias_variable(shape):
		initial = tf.constant(0.1 , shape = shape)
		return tf.Variable(initial)

def conv2d_p(x ,W  ,s):
		return tf.layers.conv2d(x, W, strides=[1,s,s,1] , padding = 'SAME')

def conv2d_1(x , W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1] , padding = 'SAME')

def max_pool_3x3(x):
		#return tf.nn.max_pool(x,  ksize=[1,2,2,1] , strides=[1,2,2,1] , padding = "VALID")
		return tf.nn.max_pool(x,  ksize=[1,2,2,1] , strides=[1,2,2,1] , padding = "SAME")

def norm(x):
		return tf.nn.lrn(x , 4 , bias = 1.0 , alpha = 0.001/9.0 , beta = 0.75)



def dropout(x):
		return(tf.nn.dropout(x , keep_prob))

def softmax(x, W, b):
		return(tf.nn.softmax(tf.matmul(x, W) + b))

def loss_cross_entropy(x, y):
		return(-tf.reduce_sum(x * tf.log(y)))
		#return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x , labels = y)))




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
		W_conv1  = weight_variable([11,11,1,64])
		b_conv1  = bias_variable([64])
		#h_conv1 = tf.nn.relu(conv2d_p(x_image, W_conv1 ,  4) + b_conv1)
		#h_conv1 = tf.nn.relu(conv2d_p(x_image, weights['wc1'] ,  4) + biases['bc1'])
		#h_conv1  = tf.nn.relu(conv2d_1(x_image, weights['wc1']) + biases['bc1'])
		h_conv1  = conv2d_1(x_image, mnist_weights['wc1']) + mnist_biases['bc1']
	
		#=============================batch normalization
		bn_conv1 = tf.layers.batch_normalization(h_conv1, training=is_training, name='bn1')
		bn_conv1 = tf.nn.relu(bn_conv1)
		
		
		print('\n*******h_conv1 is: ' + str(h_conv1))
		print('\n*******h_conv1 type is: ' + str(type(h_conv1)))
		h_pool1 = max_pool_3x3(bn_conv1)



		print('\n*******h_pool1 is: ' + str(h_pool1))
		print('\n*******h_pool1 type is: ' + str(type(h_pool1)))

		#=============================remove LRN
		h_norm1 = norm(h_pool1)

		#layer 2
		W_conv2 = weight_variable([5,5,64,192])
		b_conv2 = bias_variable([192])
		#h_conv2 = tf.nn.relu(conv2d_p(h_norm1, W_conv2 ,  1) + b_conv2)
		#h_conv2 = tf.nn.relu(conv2d_p(h_norm1, weights['wc2'] ,  1) + biases['bc2'])
		
		
		#h_conv2 = tf.nn.relu(conv2d_1(h_pool1, weights['wc2'] ) + biases['bc2'])
		h_conv2  = conv2d_1(h_norm1, mnist_weights['wc2']) + mnist_biases['bc2']
	
		#=============================batch normalization
		bn_conv2 = tf.layers.batch_normalization(h_conv2, training=is_training, name='bn2')
		bn_conv2 = tf.nn.relu(bn_conv2)



		print('\n*******h_conv2 is: ' + str(h_conv2))
		print('\n*******h_conv2 type is: ' + str(type(h_conv2)))
		h_pool2 = max_pool_3x3(bn_conv2)
		
		print('\n*******h_pool2 is: ' + str(h_pool2))
		print('\n*******h_pool2 type is: ' + str(type(h_pool2)))




		#=============================remove LRN
		h_norm2 = norm(h_pool2)

		############################
		#layer 3
		W_conv3 = weight_variable([3,3,192,384])
		b_conv3 = bias_variable([384])
		#h_conv3 = tf.nn.relu(conv2d_1(h_conv2, W_conv3 ) + b_conv3)   
		# h_conv2----> ?,14,14,384
		# h_pool2----> ?,7,7,384
		h_conv3 = conv2d_1(h_norm2, mnist_weights['wc3'] ) + mnist_biases['bc3']
		bn_conv3 = tf.layers.batch_normalization(h_conv3, training=is_training, name='bn3')
		bn_conv3 = tf.nn.relu(bn_conv3)
		
		
		h_norm3 = norm(bn_conv3)

		print('\n*******h_conv3 is: ' + str(h_conv3))
		print('\n*******h_conv3 type is: ' + str(type(h_conv3)))
		W_conv4 = weight_variable([3,3,384,384])
		b_conv4 = bias_variable([384])
		#h_conv4 = tf.nn.relu(conv2d_1(h_conv3, W_conv4 ) + b_conv4)
		h_conv4 = conv2d_1(h_norm3, mnist_weights['wc4'] ) + mnist_biases['bc4']
		bn_conv4 = tf.layers.batch_normalization(h_conv4, training=is_training, name='bn4')
		bn_conv4 = tf.nn.relu(bn_conv4)

		
		
		h_norm4 = norm(bn_conv4)

		print('\n*******h_conv4 is: ' + str(h_conv4))
		print('\n*******h_conv4 type is: ' + str(type(h_conv4)))

		W_conv5 = weight_variable([3,3,384,256])
		b_conv5 = bias_variable([256])
		#h_conv5 = tf.nn.relu(conv2d_1(h_conv4, W_conv5 ) + b_conv5)

		h_conv5 = conv2d_1(h_norm4, mnist_weights['wc5'] ) + mnist_biases['bc5']
		bn_conv5 = tf.layers.batch_normalization(h_conv5, training=is_training, name='bn5')
		bn_conv5 = tf.nn.relu(bn_conv5)

		
		#h_conv5 = tf.nn.relu(conv2d_p(h_norm4, weights['wc5'] , 2) + biases['bc5'])
		
		h_pool5 = max_pool_3x3(h_conv5)
		h_norm5 = norm(h_pool5)


		print('\n*******h_conv5 is: ' + str(h_conv5))
		print('\n*******h_conv5 type is: ' + str(type(h_conv5)))

		print('\n*******h_pool5 is: ' + str(h_pool5))
		print('\n*******h_pool5 type is: ' + str(type(h_pool5)))


		#layer 4
		W_fc1 = weight_variable([4*4*256 , 4096])
		b_fc1 = bias_variable([4096])
		#h_conv6 = tf.reshape(h_norm5 , [-1, 4*4*256])

		h_conv6 = tf.reshape(h_norm5, [-1, mnist_weights['wd1'].get_shape().as_list()[0]])

		#print('\n*******h_conv5 is: ' + str(h_conv5))
		#print('\n*******h_conv5 type is: ' + str(type(h_conv5)))
	   
		print('\n*******h_conv6 is: ' + str(h_conv6))
		print('\n*******h_conv6 type is: ' + str(type(h_conv6)))

		#h_fc1 = tf.nn.relu(tf.matmul(h_conv6, W_fc1) + b_fc1)
	   
		h_fc1  = tf.matmul(h_conv6, mnist_weights['wd1']) + mnist_biases['bd1']
		bn_fc1 = tf.layers.batch_normalization(h_fc1, training=is_training, name='fc1')
		bn_fc1 = tf.nn.relu(bn_fc1)

		
		h_fc1 = dropout(bn_fc1)


		print('\n*******h_fc1 is: ' + str(h_fc1))
		print('\n*******h_fc1 type is: ' + str(type(h_fc1)))



		W_fc2 = weight_variable([4096 , 4096])
		b_fc2 = bias_variable([4096])
		#h_fc2 = tf.nn.relu(tf.matmu(h_fc1, W_fc2) + b_fc2)
		#h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
		#h_fc2 = tf.nn.relu(tf.matmul(h_fc1, weights['wd2']) + biases['bd2'])
		
		
		h_fc2  = tf.matmul(h_fc1, mnist_weights['wd2']) + mnist_biases['bd2']
		bn_fc2 = tf.layers.batch_normalization(h_fc2, training=is_training, name='fc2')
		bn_fc2 = tf.nn.relu(bn_fc2)




		h_fc2 = dropout(bn_fc2)
		print('\n*******h_fc2 is: ' + str(h_fc2))
		print('\n*******h_fc2 type is: ' + str(type(h_fc2)))


		W_fc3 = weight_variable([4096 , 10])
		b_fc3 = bias_variable([10])
		#h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
		#h_fc3 = tf.nn.softmax(tf.matmul(h_fc2 , weights['out']) + biases['out'])
		h_fc3 = tf.matmul(h_fc2 , mnist_weights['out']) + mnist_biases['out']
		print('\n*******h_fc3 is: ' + str(h_fc3))
		print('\n*******h_fc3 type is: ' + str(type(h_fc3)))

		#h_fc3 = dropout(h_fc3)

		#loss   = loss_cross_entropy(h_fc3 , mnist_y)


		loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h_fc3 , labels = mnist_y))


		print('loss is :' + str(loss))
		print('loss type is :' + str(type(loss)))
		
		optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(mnist_y , 1))
		accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))


		saver = tf.train.Saver()
		print('\n+++++++++++ saver is: ' + str(saver) + '\n')

		return(h_fc3)

#-----------training start -----------#
def cnn_train_mnist():
	with tf.Session(config=tf.ConfigProto(
					device_count={"CPU":3},
					inter_op_parallelism_threads=1,
					intra_op_parallelism_threads=1,
					)) as sess:
		train_acc = []
		learning_rate = learning_rate_init
		indice = range(0, 5000 ,10)
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		
		print('mnist.test.images length is: ' + str(len(mnist.test.images)))
		
		
		
		for i in range(1000000):
				#begin_1 = time.time()

				
				batch = mnist.train.next_batch(batch_size)

				#batch = mnist.train.next_batch(batch_size)
				'''                        
				print('\n^^^^^^^^batch[0] is :' + str(batch[0]) + '\n')
				print('\n^^^^^^^^batch[1] is :' + str(batch[1]) + '\n')
				print('\n^^^^^^^^batch[0] length is :' + str(len(batch[0])) + '\n')
				print('\n^^^^^^^^batch[1] length is :' + str(len(batch[1])) + '\n')

				print('\n^^^^^^^^accuracy is :' + str(accuracy) + '\n')
				#print('\n^^^^^^^^accuracy length is :' + str(len(accuracy)) + '\n')
				'''
				#if i%50 == 0:
				optimizer_1.run(feed_dict = {mnist_x:batch[0] , mnist_y:batch[1] , keep_prob:0.5})
				if i%5 == 0:
						#begin_2 = time.time()
					   


						loss_1 = sess.run(loss, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})
						 
						#train_accuracy = accuracy.eval(feed_dict={mnist_x:batch[0] , mnist_y:batch[1], keep_prob:1.0})
						train_accuracy = sess.run(accuracy, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})
						
						train_acc.append(train_accuracy)
						#print ("step %d, training accuracy %g" % (i, train_accuracy))
						print ("Iter " + str((i+1)*batch_size) + ", Minibatch Loss = " + str(loss_1) + ", Training Accuracy = " + str(train_accuracy)  + ",  learning rate = " + str(learning_rate))


						# save model for reusing
						
						saver.save(sess, './../model/model_alexnet_mnist.ckpt')
						#end_2 = time.time()

						#print('Using time 2 :' + str(end_2 - begin_2) + '\n')
				
				if i == 2000:
						sess.close()
						break
				


				if i<100:
						learning_rate = learning_rate_init
				elif i>100 and i<200:
						learning_rate = learning_rate_init / 10
				
				elif i>200 and i<500:
						learning_rate = learning_rate_init / 100
				
				elif i>500 and i<1000:
						learning_rate = learning_rate_init / 1000

				else:
						learning_rate = 0.0000001


				#end_1 = time.time()
				#print('Using time 1 :' + str(end_1 - begin_1) + '\n')
		








        
'''
				#print('optimizer is: ' + str(optimizer))
				plt.plot(indice, train_acc, 'k-', label = "Train Set Accuracy")
				plt.title('Train Accuracy')
				plt.xlabel('Generation')
				plt.ylabel('Accuracy')
				plt.show()


def cnn_train():
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				# Keep training until reach max iterations
				#while step * batch_size < training_iters:
				for i in range(5000):
						batch_xs, batch_ys = mnist.train.next_batch(batch_size)

						sess.run(optimizer_1, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 0.75})
				   
						if i % 5 == 0:
								acc = sess.run(accuracy, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 1.})
								

								loss_1 = sess.run(loss, feed_dict={mnist_x: batch_xs, mnist_y: batch_ys, keep_prob: 1.})
								






								print('loss_1 is: ' + str(loss_1))
								print('loss_1 type is: ' + str(type(loss_1)))
								print ("Iter " + str(i) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(acc))
				
				print ("Optimization Finished!")
				print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))





'''

#predict
def predict_mnist():
		
		#mnist test set 有10000张图片
		#2500张图片已经内存溢出

		
		batch_size = 2000
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		#saver.restore(sess , 'model')
		saver.restore(sess, './../model/model_alexnet_mnist.ckpt')

		#mnist 的test batch
		test_batch    = mnist.test.next_batch(batch_size)

		
		loss_2        = sess.run(loss,     feed_dict={mnist_x: test_batch[0], mnist_y: test_batch[1], keep_prob: 1.})
		test_accuracy = sess.run(accuracy, feed_dict={mnist_x: test_batch[0], mnist_y: test_batch[1], keep_prob: 1.})
		

		#for i in range(5):
		print ("Iter " + str(batch_size) + ", Minibatch Loss = " + str(loss_2) + ", Testing Accuracy = " + str(test_accuracy) )
		
		
		
		#print( "test accuracy %g" % accuracy.eval(feed_dict={mnist_x:mnist.test.images, mnist_y:mnist.test.labels, keep_prob:1.0}))





if __name__ == '__main__':
		working_flow_mnist()
		

		'''
		loss   = loss_cross_entropy(h_fc3 , mnist_y)
		optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(mnist_y , 1))
		accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))
		'''
	   


		#tf.reset_default_graph() 
		saver = tf.train.Saver()
		print('\n+++++++++++ saver is: ' + str(saver) + '\n')

		cnn_train_mnist()
		predict_mnist()




















