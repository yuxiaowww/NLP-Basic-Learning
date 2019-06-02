# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:39:35 2019

@author: yuwei
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

'sklearn：朴素贝叶斯（naïve beyes）'

#%% 
'高斯模型'
#数据集准备
iris = datasets.load_iris()
train_data,train_label = iris.data,iris.target
#高斯模型
clf = GaussianNB()
clf.fit(train_data, train_label)
#新样本预测
print(clf.predict([iris.data[2]]))
#预测训练集
pre = clf.predict(train_data)
#混淆矩阵计算
print(confusion_matrix(train_label, pre))


#%% 
'多项式模型'
#数据集准备
iris = datasets.load_iris()
train_data,train_label = iris.data,iris.target
#多项式模型
clf = MultinomialNB()
clf.fit(train_data, train_label)
#新样本预测
print(clf.predict([iris.data[2]]))
#预测训练集
pre = clf.predict(train_data)
#混淆矩阵计算
print(confusion_matrix(train_label, pre))


#%% 
'伯努利模型'
#数据集准备
iris = datasets.load_iris()
train_data,train_label = iris.data,iris.target
#伯努利模型
clf = BernoulliNB()
clf.fit(train_data, train_label)
#新样本预测
print(clf.predict([iris.data[2]]))
#预测训练集
pre = clf.predict(train_data)
#混淆矩阵计算
print(confusion_matrix(train_label, pre))

