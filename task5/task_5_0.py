# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:16:10 2019

@author: yuwei
"""

'https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=5&sn=25e11ef5f6e014647af1061631047521'

import pandas as pd

'正则化'
from keras.models import Sequential
from keras.layers import Dense, Embedding

#%% 
'1-L2 & L1 正则化'
'''
不同于L2，权重值可能被减少到0。
因此，L1对于压缩模型很有用。
其它情况下，一般选择优先选择L2正则化。
'''
from keras import regularizers
# 在Dense层应用L2正则化
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01)))


#%%
'2-Dropout'
'''
Dropout的原理很简单：
在每个迭代过程中，随机选择某些节点，并且删除前向和后向连接
具有较大的神经网络时，通常首选dropout以引入更多的随机性
'''
from keras.layers.core import Dropout
model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(0.25),
Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
 ])


#%%
'3-数据扩增'
'''
减少过拟合的最简单方法是增加训练样本。
在机器学习中，由于标注数据是昂贵的，我们不能够增加训练样本数量。
但是对于图像问题，
有几种可以增加训练样本的方法-旋转（rotaing）、
翻转（flipping）、放缩（scaling）及平移（shfiting）等。
'''
#ImageDataGenerator来实现上述的图像变换
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(train)


#%%
'4-早期停止'
'''
早期停止（early stopping）是一种交叉验证策略，
我们将一部分训练集作为验证集（validation set）。 
当我们看到验证集的性能越来越差时，我们立即停止对该模型的训练。
这被称为早期停止。
'''
#在Keras中，我们可以使用callbacks函数实现早期停止
#patience参数epochs数量，当在这个过程性能无提升时会停止训练。
from keras.callbacks import EarlyStopping
EarlyStopping(monitor='val_err', patience=5)












