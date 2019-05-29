# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:57:55 2019

@author: yuwei
"""

from tensorflow import keras
from collections import Counter
import numpy as np
import pandas as pd

TRAIN_PATH = 'cnews/cnews.train.txt'
VAL_PATH = 'cnews/cnews.val.txt'
TEST_PATH = 'cnews/cnews.test.txt'
VOCAB_SIZE = 5000
MAX_LEN = 600
BATCH_SIZE = 64


def read_file(filename):
    
    "读取数据集"
    
    contents,labels = [],[]
    file_path = {'train':TRAIN_PATH,'val':VAL_PATH,'test':TEST_PATH}
    
    with open(file_path[filename],'r',encoding='utf-8') as f:
        for line in f:
            try:
                #strip()函数,去除字符串首尾的空格;split()函数,空格切分
                labels.append(line.strip().split('\t')[0])
                contents.append(line.split('\t')[1])
            except:
                pass
        data = pd.DataFrame()
        data['text'] = contents
        data['label'] = labels
    
    return data

def build_vocab(data):
    
    "构建词汇表"
    
    #使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理
    all_content = []
    #遍历函数iterrows(),index和内容
    for _,text in data.iterrows():
        #extend()区别与append()函数,是将list中每个元素逐个添加
        all_content.extend(text['text'])
    #计数,返回字典
    counter = Counter(all_content)
    #most_common([n]),返回一个TopN列表.如果n没有被指定,则返回所有元素
    count_pairs = counter.most_common(VOCAB_SIZE - 1)
    #仅取字出来
    words = [i[0] for i in count_pairs]
    words = ['<PAD>'] + list(words)
    
    return words
    
def read_vocab(words):
    
    "读取上一步存储的词汇表,转换为{词：id}表示"
    
    #zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
    #dict()函数强转为字典,单词对应于id
    words_id = dict(zip(words, range(len(words))))
    
    return words_id   

def read_category(data):
   
    "将分类目录固定，转换为{类别: id}表示 "
    
    category = list(set(data['label']))
    category_id = dict(zip(category, range(len(category))))
    
    return category_id

def to_words(content, words):
    
    "将一条由id表示的数据重新转换为文字"
    
    words_ = ' '.join(words[i] for i in content)
    
    return words_

def process_file(data, words_id, category_id):
    
    "将数据集从文字转换为固定长度的id序列表示"
    
    content = data['text']
    labels = data['label']
    content_id,label_id = [],[]
    for text, label in zip(content,labels):
        content_id.append([words_id[i] for i in text if i in words_id])
        label_id.append(category_id[label])
    

    
    return content_id, label_id    

def batch_iter(x, y):
    
    "为神经网络的训练准备经过shuffle的批次的数据"
    
    num_batch = int((len(x) - 1) / BATCH_SIZE) + 1
    #permutation()随机排列一个数组
    indices = np.random.permutation(np.arange(len(x)))
    #保证每次可以随机选取一个batch_size的数据出来
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * BATCH_SIZE
        end_id = min((i + 1) * BATCH_SIZE, len(x))
        #yield关键字的核心用法,即逐个生成,避免浪费内存
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id] 
        print(x_shuffle[start_id:end_id])

if __name__ == '__main__':
    "主函数入口"
    
    #获取数据集
    train = read_file('train')
    test = read_file('test')
    val = read_file('val')
    #打印标签类别
    print(set(train['label']))
    words = build_vocab(train)
    words_id = read_vocab(words)
    category_id = read_category(train)
    x_pad, y_pad = process_file(train, words_id, category_id)



















