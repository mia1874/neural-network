# -*- coding=utf-8 -*-
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./../data/MNIST_data", one_hot=True)

import tensorflow as tf
# 定义网络超参数
learning_rate = 0.001
training_iters = 5000
batch_size = 64
#display_step = 20
display_step = 2

# 定义网络参数
n_input = 784 # 输入的维度
n_classes = 10 # 标签的维度
dropout = 0.75 # Dropout 的概率

# 占位符输入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 卷积操作
def conv2d(name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

# 最大下采样操作
def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# 存储所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([192])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
# 定义整个网络
def alex_net(_X, _weights, _biases, _dropout):
        # 向量转为矩阵
        _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

        # 第一层卷积
        # 卷积
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # 下采样
        pool1 = max_pool('pool1', conv1, k=2)
        # 归一化
        norm1 = norm('norm1', pool1, lsize=4)
        print('\n*******conv1 is:' + str(conv1))
        print('\n*******conv1 type is:' + str(type(conv1)))
        print('\n*******pool1 is:' + str(pool1))
        print('\n*******pool1 type is:' + str(type(pool1)))
        print('\n*******norm1 is:' + str(norm1))
        print('\n*******norm1 type is:' + str(type(norm1)))

        # 第二层卷积
        # 卷积
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # 下采样
        pool2 = max_pool('pool2', conv2, k=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=4)
        print('\n*******conv2 is:' + str(conv2))
        print('\n*******conv2 type is:' + str(type(conv2)))
        print('\n*******pool2 is:' + str(pool2))
        print('\n*******pool2 type is:' + str(type(pool2)))
        print('\n*******norm2 is:' + str(norm2))
        print('\n*******norm2 type is:' + str(type(norm2)))


        #############################
        # 第三层卷积
        # 卷积
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # 归一化
        norm3 = norm('norm3', conv3, lsize=4)
        print('\n*******conv3 is:' + str(conv3))
        print('\n*******conv3 type is:' + str(type(conv3)))
        print('\n*******norm3 is:' + str(norm3))
        print('\n*******norm3 type is:' + str(type(norm3)))

        # 第四层卷积
        # 卷积
        conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
        # 归一化
        norm4 = norm('norm4', conv4, lsize=4)
        print('\n*******conv4 is:' + str(conv4))
        print('\n*******conv4 type is:' + str(type(conv4)))
        print('\n*******norm4 is:' + str(norm4))
        print('\n*******norm4 type is:' + str(type(norm4)))

        # 第五层卷积
        # 卷积
        conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
        print('\n*******conv5 is:' + str(conv5))
        print('\n*******conv5 type is:' + str(type(conv5)))
        

        # 下采样
        pool5 = max_pool('pool5', conv5, k=2)
        print('\n*******pool5 is:' + str(pool5))
        print('\n*******pool5 type is:' + str(type(pool5)))
        

        # 归一化
        norm5 = norm('norm5', pool5, lsize=4)
        print('\n*******norm5 is:' + str(norm5))
        print('\n*******norm5 type is:' + str(type(norm5)))



        # 全连接层1，先把特征图转为向量
        dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
        print('\n*******dense1  1  is:' + str(dense1))
        print('\n*******dense1  1  type is:' + str(type(dense1)))
       
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
        print('\n*******dense1  2  is:' + str(dense1))
        print('\n*******dense1  2  type is:' + str(type(dense1)))
        
        dense1 = tf.nn.dropout(dense1, _dropout)
        print('\n*******dense1  3  is:' + str(dense1))
        print('\n*******dense1  3  type is:' + str(type(dense1)))

        # 全连接层2
        dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])
        print('\n*******dense2  1  is:' + str(dense2))
        print('\n*******dense2  1  type is:' + str(type(dense2)))
        
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
        print('\n*******dense2  2  is:' + str(dense2))
        print('\n*******dense2  2  type is:' + str(type(dense2)))

        dense2 = tf.nn.dropout(dense2, _dropout)
        print('\n*******dense2  3  is:' + str(dense2))
        print('\n*******dense2  3  type is:' + str(type(dense2)))

        # 网络输出层
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
        print('\n*******out  is:' + str(out))
        print('\n*******out  type is:' + str(type(out)))
        return out
 
# 构建模型
pred = alex_net(x, weights, biases, keep_prob)
 
# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
# 准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
# 初始化所有的共享变量
init = tf.initialize_all_variables()
 
# 开启一个训练
with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        #while step * batch_size < training_iters:
        for i in range(5000): 
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                '''
                print('\n*****************batch_xs is : ' + str(batch_xs) + '\n')
                print('\n*****************batch_ys is : ' + str(batch_ys) + '\n')
                print('\n*****************batch_xs length is : ' + str(len(batch_xs)) + '\n')
                print('\n*****************batch_ys length is : ' + str(len(batch_ys)) + '\n')

                print('\n&&&&&&&&&&&&&&&&&  accuracy is : ' + str(accuracy) + '\n')
                #print('\n&&&&&&&&&&&&&&&&&  accuracy length is : ' + str(len(accuracy)) + '\n')
                '''

                # 获取批数据
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                #if step % display_step == 0:
                if i % 5 == 0 :
                        # 计算精度
                        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                        # 计算损失值
                        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                        #print ("Iter " + str(step*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
                        #if i % 5 == 0 :
                        print ("Iter " + str(i) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
                step += 1
        print ("Optimization Finished!")
        # 计算测试精度
        print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))


