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


file_info = (
        '''
        original code for AlexNet
        Using for MNIST , Cifar-10

        Author: MrZQ


        v1.0.0.1003_alpha Update
                1. AlexNet working on MNIST data set
                2. Bug fix-

        v1.0.1.1008_alpha Update
                1. Working well on MNIST data set
                2. Fix bug for 'loss is Nan'
                3. ready for adding cifar10 data set




        ''')







import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time
import cv2
import pickle
import sys
import os


#-------------variable init-----------------#
keep_prob = tf.placeholder("float")
#keep_prob = 0.75
#learning_rate = 0.0001-----> 0.1
#learning_rate = 0.01  -----> 0.2 0.03
learning_rate = 0.001
accuracy = []
saver = []
optimizer_1 = []
loss = []




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
        'out': tf.Variable(tf.random_normal([10]))
}


#-------------data set import-----------------#

mnist = input_data.read_data_sets('./../data/MNIST_data' , one_hot = True)
mnist_x  = tf.placeholder("float", shape = [None, 784])
mnist_y  = tf.placeholder("float", shape = [None, 10])





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


print('MNIST is: ' + str(mnist))
print('mnist[0] is: ' + str(mnist[0]))





#-------------function init-----------------#
def weight_variable(shape):
        initial = tf.truncated_normal(shape , stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0.1 , shape = shape)
        return tf.Variable(initial)

def conv2d_p(x ,W  ,s):
        return tf.nn.conv2d(x, W, strides=[1,s,s,1] , padding = 'SAME')

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
        #return(-tf.reduce_sum(x * tf.log(y)))
        return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x , labels = y)))




def optimizer(learning_rate, cross_entropy):
        return(tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy))

def correct_predict(a,b):
        return(tf.equal(tf.argmax(a,1) , tf.argmax(b,1)))

def accuracy(predict):
        return(tf.reduce_mean(tf.cast(predict, "float")))


def mnist_2_figure(x):
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

def working_flow():
        global accuracy
        global saver
        global optimizer_1
        global loss

        #layer 0
        x_image = tf.reshape(mnist_x, [-1,28,28,1])


        #layer 1
        W_conv1 = weight_variable([11,11,1,64])
        b_conv1 = bias_variable([64])
        #h_conv1 = tf.nn.relu(conv2d_p(x_image, W_conv1 ,  4) + b_conv1)
        #h_conv1 = tf.nn.relu(conv2d_p(x_image, weights['wc1'] ,  4) + biases['bc1'])
        h_conv1 = tf.nn.relu(conv2d_1(x_image, weights['wc1']) + biases['bc1'])


        print('\n*******h_conv1 is: ' + str(h_conv1))
        print('\n*******h_conv1 type is: ' + str(type(h_conv1)))
        h_pool1 = max_pool_3x3(h_conv1)



        print('\n*******h_pool1 is: ' + str(h_pool1))
        print('\n*******h_pool1 type is: ' + str(type(h_pool1)))


        h_norm1 = norm(h_pool1)

        #layer 2
        W_conv2 = weight_variable([5,5,64,192])
        b_conv2 = bias_variable([192])
        #h_conv2 = tf.nn.relu(conv2d_p(h_norm1, W_conv2 ,  1) + b_conv2)
        #h_conv2 = tf.nn.relu(conv2d_p(h_norm1, weights['wc2'] ,  1) + biases['bc2'])
        h_conv2 = tf.nn.relu(conv2d_1(h_norm1, weights['wc2'] ) + biases['bc2'])

        print('\n*******h_conv2 is: ' + str(h_conv2))
        print('\n*******h_conv2 type is: ' + str(type(h_conv2)))
        h_pool2 = max_pool_3x3(h_conv2)

        print('\n*******h_pool2 is: ' + str(h_pool2))
        print('\n*******h_pool2 type is: ' + str(type(h_pool2)))

        h_norm2 = norm(h_pool2)

        ############################
        #layer 3
        W_conv3 = weight_variable([3,3,192,384])
        b_conv3 = bias_variable([384])
        #h_conv3 = tf.nn.relu(conv2d_1(h_conv2, W_conv3 ) + b_conv3)
        # h_conv2----> ?,14,14,384
        # h_pool2----> ?,7,7,384
        h_conv3 = tf.nn.relu(conv2d_1(h_norm2, weights['wc3'] ) + biases['bc3'])
        h_norm3 = norm(h_conv3)

        print('\n*******h_conv3 is: ' + str(h_conv3))
        print('\n*******h_conv3 type is: ' + str(type(h_conv3)))
        W_conv4 = weight_variable([3,3,384,384])
        b_conv4 = bias_variable([384])
        #h_conv4 = tf.nn.relu(conv2d_1(h_conv3, W_conv4 ) + b_conv4)
        h_conv4 = tf.nn.relu(conv2d_1(h_norm3, weights['wc4'] ) + biases['bc4'])
        h_norm4 = norm(h_conv4)

        print('\n*******h_conv4 is: ' + str(h_conv4))
        print('\n*******h_conv4 type is: ' + str(type(h_conv4)))

        W_conv5 = weight_variable([3,3,384,256])
        b_conv5 = bias_variable([256])
        #h_conv5 = tf.nn.relu(conv2d_1(h_conv4, W_conv5 ) + b_conv5)
        #following is old:
        h_conv5 = tf.nn.relu(conv2d_1(h_conv4, weights['wc5'] ) + biases['bc5'])
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
        h_conv6 = tf.reshape(h_norm5, [-1, weights['wd1'].get_shape().as_list()[0]])


        #print('\n*******h_conv5 is: ' + str(h_conv5))
        #print('\n*******h_conv5 type is: ' + str(type(h_conv5)))

        print('\n*******h_conv6 is: ' + str(h_conv6))
        print('\n*******h_conv6 type is: ' + str(type(h_conv6)))

        #h_fc1 = tf.nn.relu(tf.matmul(h_conv6, W_fc1) + b_fc1)

        h_fc1 = tf.nn.relu(tf.matmul(h_conv6, weights['wd1']) + biases['bd1'])
        h_fc1 = dropout(h_fc1)



        print('\n*******h_fc1 is: ' + str(h_fc1))
        print('\n*******h_fc1 type is: ' + str(type(h_fc1)))



        W_fc2 = weight_variable([4096 , 4096])
        b_fc2 = bias_variable([4096])
        #h_fc2 = tf.nn.relu(tf.matmu(h_fc1, W_fc2) + b_fc2)
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, weights['wd2']) + biases['bd2'])
        h_fc2 = dropout(h_fc2)
        print('\n*******h_fc2 is: ' + str(h_fc2))
        print('\n*******h_fc2 type is: ' + str(type(h_fc2)))


        W_fc3 = weight_variable([4096 , 10])
        b_fc3 = bias_variable([10])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        #h_fc3 = tf.nn.softmax(tf.matmul(h_fc2 , weights['out']) + biases['out'])
        h_fc3 = tf.matmul(h_fc2 , weights['out']) + biases['out']
        print('\n*******h_fc3 is: ' + str(h_fc3))
        print('\n*******h_fc3 type is: ' + str(type(h_fc3)))

        #h_fc3 = dropout(h_fc3)

        loss   = loss_cross_entropy(h_fc3 , mnist_y)

        optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(mnist_y , 1))
        accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))


        saver = tf.train.Saver()
        print('\n+++++++++++ saver is: ' + str(saver) + '\n')

        return(h_fc3)

#-----------training start -----------#


def cnn_train():
        with tf.Session() as sess:
                train_acc = []
                #indice = range(0, 5000 ,10)
                sess = tf.InteractiveSession()
                sess.run(tf.global_variables_initializer())
                #sess.run(tf.initialize_all_variables())
                for i in range(100):
                        #begin_1 = time.time()
                        batch = mnist.train.next_batch(64)


                        batch_1 = load_CIFAR_batch('./../data/cifar-10-batches-py/data_batch_1')

                        print(batch_1[0][0])

                        plt.imshow(np.reshape((batch_1[0][i]/256), [32 , 32 , 3]) )
                        #plt.imshow(np.reshape((batch_1[0][i]), [32 , 32 , 3]) )



                        plt.show()

                        """

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
                        """
                        #if i%50 == 0:
                        if i%5 == 0:
                                begin_2 = time.time()

                                #train_accuracy = accuracy.eval(feed_dict={mnist_x:batch[0] , mnist_y:batch[1], keep_prob:1.0})
                                train_accuracy = sess.run(accuracy, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})

                                loss_1 = sess.run(loss, feed_dict={mnist_x: batch[0], mnist_y: batch[1], keep_prob: 1.})


                                train_acc.append(train_accuracy)
                                #print ("step %d, training accuracy %g" % (i, train_accuracy))
                                # save model for reusing

                                saver.save(sess, './../model/model_alexnet_mnist.ckpt')
                                end_2 = time.time()

                                #print('Using time 2 :' + str(end_2 - begin_2) + '\n')


                                print ("Iter " + str((i+1)*64) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy))


                                #plt.title('Actual: ' + mnist_2_figure(str(batch[1][0])) + '        Predict: ' + mnist_2_figure(str(batch[1][i])) + '\n' + "Iter " + str((i+1)*64) + ", Minibatch Loss = " + "{:.6f}".format(loss_1) + ", Training Accuracy = " + "{:.5f}".format(train_accuracy))
                                #plt.title('Actual: ' + mnist_2_figure(str(batch[1][0])) + '        Predict: ' + mnist_2_figure(str(batch[1][i])) )
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
        #saver.restore(sess , 'model')
        saver.restore(sess, './../model/model_alexnet_mnist.ckpt')
        print( "test accuracy %g" % accuracy.eval(feed_dict={mnist_x:mnist.test.images, mnist_y:mnist.test.labels, keep_prob:1.0}))





if __name__ == '__main__':
        working_flow()


        '''
        loss   = loss_cross_entropy(h_fc3 , mnist_y)
        optimizer_1     = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        correct_predict = tf.equal(tf.argmax(h_fc3 , 1) , tf.argmax(mnist_y , 1))
        accuracy        = tf.reduce_mean(tf.cast(correct_predict, "float"))
        '''

        saver = tf.train.Saver()
        print('\n+++++++++++ saver is: ' + str(saver) + '\n')

        cnn_train()
        predict()
