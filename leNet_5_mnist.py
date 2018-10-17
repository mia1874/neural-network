#-*- coding : utf-8 -*-

'''#------------------------------------------#
LeNet-5 :

Type:           Maps    Size    Kernelsize Stride

input            1      32x32   -            -
Conv             6      28x28   5x5          1
AvgPooling       6      14x14   2x2          2
Conv             16     10x10   5x5          1
AvgPooling       16     5x5     2x2          2
fully connected  120    1x1     5x5          1
fully connected  -      84      -            -
fully connected  -      10      -            -

#------------------------------------------#'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


#----------- variable init ------------#
'''
#----- old version start ------
#sess = tf.Session()
train_xdata = np.array([np.reshape(x,(28,28)) for x in mnist.train.images])
test_xdata  = np.array([np.reshape(x,(28,28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels  = mnist.test.labels
print('type train labels: '  + str(type(train_labels)))
print('train labels: '       + str(train_labels))
print('len train labels: '   + str(len(train_labels)))
#----- old version end   -------
'''


mnist = input_data.read_data_sets('./../data/MNIST_data' , one_hot = True)
x  = tf.placeholder("float", shape = [None, 784])
y_ = tf.placeholder("float", shape = [None, 10])


#-----------function init ------------#
# weight init
def weight_variable(shape):
        initial = tf.truncated_normal(shape , stddev=0.1)
        return tf.Variable(initial)

# bias init 
def bias_variable(shape):
        initial = tf.constant(0.1 , shape = shape)
        return tf.Variable(initial)

# convolution init 
# x--->四维张量 ---->[batch , height, width, channels]
# strides--->height width 步长 2 2
# padding--->same
def conv2d(x , W):
        return tf.nn.conv2d(x, W, strides = [1,1,1,1] , padding = 'SAME')


#pooling init 
#平均池化，取平均值作为结果
# x------>四维张量 ---->[batch , height, width, channels]
# ksize-->pool窗口2x2
# strides--->height width 步长 2 2 
# padding--->same
def avg_pool_2x2(x):
        return tf.nn.avg_pool(x, ksize=[1,2,2,1] , strides=[1,2,2,1] ,padding = "SAME")


#----------- network init ------------#
# No.0 :  input
# No.1 :  conv---->pooling
# No.2 :  conv---->pooling
# No.3 :  conv---->pooling (not ready)

#No.1 Layer     conv + pooling
#初始化W为[5,5,1,6]的张量，表示卷积核大小为5*5，1表示图像通道数，6表示卷积核个数即输出6个特征图
W_conv1 = weight_variable([5,5,1,6])
#初始化b为[6],即输出大小
b_conv1 = bias_variable([6])

#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
#-1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,6]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = avg_pool_2x2(h_conv1)



#No.2 Layer     conv + pooling 
#卷积核大小依然是5*5，通道数为6，卷积核个数为16
W_conv2 = weight_variable([5,5,6,16])
b_conv2 = weight_variable([16])

#h_pool2即为第二层网络输出，shape为[batch,7,7,16]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_2x2(h_conv2)



#No.3 Layer，   fully connected
W_fc1 = weight_variable([7*7*16 , 120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2 , [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#dropout Layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)


#Output Layer           ： softmax
W_fc2 = weight_variable([120 , 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop , W_fc2) + b_fc2)


#Loss Layer             : cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))


#Gradient Decent Layer  : ADAM,   learning rate 0.0001--->le-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#Accuracy Layer
correct_predict = tf.equal(tf.argmax(y_conv,1) , tf.argmax(y_ , 1))
accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))


saver = tf.train.Saver()

#-----------training start -----------#
def cnn_train():
        train_acc = []
        indice = range(0, 5000 ,100)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
                batch = mnist.train.next_batch(64)
                if i%10 == 0 :
                        train_accuracy = accuracy.eval(feed_dict={x:batch[0] , y_:batch[1], keep_prob:1.0})
                        train_acc.append(train_accuracy)
                        print ("step %d, training accuracy %g" % (i, train_accuracy))
                        # save model for reusing
                        saver.save(sess, './../model/model_lenet5.ckpt')
                train_step.run(feed_dict = {x:batch[0] , y_:batch[1] , keep_prob:0.5})
                if i == 5000 :
                        break
        print('train_step is: ' + str(train_step))
        '''
        plt.plot(indice, train_acc, 'k-', label = "Train Set Accuracy")
        plt.title('Train Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()
        '''


        
#predict
def predict():
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        #saver.restore(sess , 'model')
        saver.restore(sess, './../model/model_lenet5.ckpt')
        print( "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
        
#cnn_train()
#predict()

if __name__ == '__main__':
        cnn_train()
        predict()











































