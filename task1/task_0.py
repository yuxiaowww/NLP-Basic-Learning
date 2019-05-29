# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:49:26 2019

@author: yuwei
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

#数据集获取
from keras.datasets import imdb
#获取数据
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
#探索数据
#格式化字符串的函数 str.format()
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
# 文本长度
print(len(train_data[0]), len(train_data[1]))


from keras.datasets import mnist
#获取minst数据-书写识别数字数据
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_labels)
