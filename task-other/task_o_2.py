# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:29:05 2019

@author: yuwei
"""

'https://www.cnblogs.com/cnXuYang/p/8992865.html'
'Keras LSTM文本分类简单示例'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  
batch_size = 32

print('下载数据...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), '训练序列')
print(len(x_test), '测试序列')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('训练数据 shape:', x_train.shape)
print('测试数据 shape:', x_test.shape)

print('构建模型...')    
model = Sequential()
#嵌入词向量
model.add(Embedding(max_features, 128))
#LSTM层
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#全连接层
model.add(Dense(1, activation='sigmoid'))
#打印构建模型的信息
model.summary()

#模型编译
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('开始训练...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=3,validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)













