# -*- encoding = utf8 -*-

import pickle
import numpy as np
import os

file_info = (
		'''
		VGG 
		Data set import function for VGG
		Using for MNIST , CIFAR-10
		Author: MrZQ
		20181019
		''')







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

			#**********************************************************************************************************************
			#label_temp = figure_2_label(train_label[j])
			#**********************************************************************************************************************

			label_temp = figure_2_label(train_label[index[j]])



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







