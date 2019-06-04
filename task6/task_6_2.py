# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:06:59 2019

@author: yuwei
"""

import fastText
 
# @param input: 训练数据文件路径
# @param lr: 学习率
# @param dim: 向量维度
# @param ws: cbow模型时使用
# @param epoch: 次数
# @param minCount: 词频阈值, 小于该值在初始化时会过滤掉
# @param minCountLabel: 类别阈值，类别小于该值初始化时会过滤掉
# @param minn: 构造subword时最小char个数
# @param maxn: 构造subword时最大char个数
# @param neg: 负采样
# @param wordNgrams: n-gram个数
# @param loss: 损失函数类型, softmax, ns: 负采样, hs: 分层softmax
# @param bucket: 词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
# @param thread: 线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
# @param lrUpdateRate: 学习率更新
# @param t: 负采样阈值
# @param label:  类别前缀
# @param verbose:
# @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
# @return model object
 
# 分类训练
classifier = fastText.train_supervised('fastText_train.txt')
 
# 模型预测，给定文本预测分类，返回预测标签和概率
label, prob = classifier.predict('6 8 5 7 1 9 7')
print(label)
print(prob)
 
# 模型预测，根据给定的数据集对模型进行评价，返回样本个数、准确率、召回率
n, accuracy, recall = classifier.test('fastText_test.txt')
print(n, accuracy, recall)





