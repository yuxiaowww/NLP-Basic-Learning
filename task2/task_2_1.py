# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:35:35 2019

@author: yuwei
"""

import jieba
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def read_stopwords(filename):
    
    "读取停用词"
    
    stopwords = []
    fp = open(filename,'r')
    for line in fp.readlines():
        stopwords.append(line.strip())
    fp.close()

    return stopwords

def cut_data(data,stopwords):
    
    "切分数据,同时删除停用词"
    
    words = []
    for text in data['text']:
        word = list(jieba.cut(text))
        for w in list(set(word) & set(stopwords)):
            while w in word:
                word.remove(w)
        words.append(' '.join(word))

    data['text'] = words
    
    return data

def word_list(data):
    
    "获取单词列表"
    
    all_words = []
    for text in data['text']:
        all_words.extend(text)
#    all_words = list(set(all_words))
    
    #对字符频率进行统计
    counter = Counter(all_words)
    count_pairs = counter.most_common()
    
    words_counter = pd.DataFrame([i[0] for i in count_pairs], columns={'words'})
    words_counter['counter'] = [i[1] for i in count_pairs]
    
    return words_counter

 
'''
CountVectorizer+TfidfTransformer
CountVectorizer会将文本中的词语转换为词频矩阵,
它通过fit_transform函数计算各个词语出现的次数,
通过get_feature_names()可获得所有文本的关键词,
通过toarray()可看到词频矩阵的结果。
'''
'''
TfidfVectorizer
将原始文档的集合转化为tf-idf特性的矩阵,
相当于CountVectorizer配合TfidfTransformer使用的效果
'''

def text_vc(data):
    
    "计算文本向量"
    count_vec = CountVectorizer(max_features=300, min_df=2)
#    count_vec = TfidfVectorizer(max_features=300, min_df=2)
    count_vec.fit_transform(data['text'])
    
    fea_vec = count_vec.transform(data['text']).toarray()
    
    return fea_vec


if __name__ == '__main__':
    
    #读入数据集
    data = pd.read_csv('cnews/cnews.train.txt',sep='\t',header=None,names=['label','text'])
    data = data.head(50)
    #读入停用词
    stop_words = read_stopwords('cnews/chinese_stop_words.txt')
    #切词
    data = cut_data(data,stop_words)
    #获取词频
    words_counter = word_list(data)
    #向量化
    fea_vec = pd.DataFrame(text_vc(data))
    


