# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:21:59 2019
@author: pc
"""

import fastText.FastText as ff
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pandas as pd

def read_file(path):
#    with open(path, 'r', encoding="UTF-8") as f:
#        data = []
#        labels = []
#        for line in f:
#            data.append(line.split('\t')[0])
#            labels.append(line.split('\t')[1])
    Data = pd.read_csv('merge.txt',sep='\t',header=None,names=['text','label'])
    Data.label = Data.label.map(lambda x:str(x))
    data = list(Data['text']);labels = list(Data['label'])
    return data, labels


def get_tokenizer_data(data):
    '''
        fasttext传入文本必须对其进行预处理和编码
    '''
    tokenizer = Tokenizer(num_words=None)
    # 得到文本的字典
    tokenizer.fit_on_texts(data)
    # 将每个string的每个词转成数字
    data = tokenizer.texts_to_sequences(data)
    return data


def fast_text_model(X_test):
    '''
        使用fasttext进行文本分类
    '''
    # 分类训练
    classifier = ff.train_supervised('train.txt', label='__label__')
    # 模型预测，返回预测标签和概率
    label, prob = classifier.predict(X_test)
    # 根据给定数据集对模型进行评价，返回样本个数、准确率、召回率
    result = classifier.test('test.txt')
    return label, prob, result
    
# 读文件
data, labels = read_file('merge.txt')

# 向量化文本
tokenizer_train = get_tokenizer_data(data)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(tokenizer_train, 
                                                    labels,
                                                    test_size = 0.2,
                                                    random_state=33)
# 在标签值前增加前缀
label_train = [('__label__' + i).replace('\n', '') for i in y_train]

# 将label列与文本列合并为一行
train = [i + ' ' + str(j).replace('[', '').replace(']', '').replace(',', '') for i, j in zip(label_train, X_train)]
    
f = open('train.txt', 'w')
for i in train:
    f.write(i)
    f.write('\n')
f.close()

label_test = [('__label__' + i).replace('\n', '') for i in y_test]
test = [i + ' ' + str(j).replace('[', '').replace(']', '').replace(',', '') for i, j in zip(label_test, X_test)]
f = open('test.txt', 'w')
for i in test:
    f.write(i)
    f.write('\n')
f.close()

X_test_tok = [str(j).replace('[', '').replace(']', '').replace(',', '') for j in X_test]

label, prob, result = fast_text_model(X_test_tok)