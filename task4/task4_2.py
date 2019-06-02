# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:50:11 2019

@author: yuwei
"""

'LDA简单示例'
'lda安装：pip install lda'
'''
p(w|d)=p(w|t)*p(t|d)  
直观的看这个公式，就是以Topic作为中间层，
可以通过当前的θd和φt给出了文档d中出现单词w的概率。
其中p(t|d)利用θd计算得到，p(w|t)利用φt计算得到。
'''

import lda
import numpy as np

#%%
'非文本简单示例'
#文档集doc,n*m的矩阵,表示有n个文本,m个单词,值表示出现次数或者是否出现
doc = np.array([[1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
model = lda.LDA(random_state=1, n_topics=3, n_iter=100)
model.fit(doc)
 
#主题-单词(topic-word)分布
topic_word = model.topic_word_
print(topic_word)
#文档主题(Document-Topic)分布
doc_topic = model.doc_topic_
print(doc_topic)

#%%
'文本简单示例'
#读入文本数据
doc = ['因为森林人即将换代，这套系统没必要装在一款即将换代的车型上，因为肯定会影响价格。',
       '斯柯达要说质量，似乎比大众要好一点，价格也低一些。我听说过野帝，但没听说过你说这车。']

#中文分词与去停用词
import jieba

def read_stopwords(filename):
    "读取停用词"
    
    stopwords = []
    fp = open(filename,'r')
    for line in fp.readlines():
        stopwords.append(line.strip())
    fp.close()

    return stopwords

def cut_data(data,stopwords):
    "切词,同时删除停用词"
    
    words = []
    for text in data:
        word = list(jieba.cut(text))
        for w in list(set(word) & set(stopwords)):
            while w in word:
                word.remove(w)
        words.append(word)

    return words

#读入停用词
stop_words = read_stopwords('chinese_stop_words.txt')
#切词
new_doc = cut_data(doc,stop_words)

#文本向量化
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(new_doc)
DT = [dictionary.doc2bow(item) for item in new_doc]

#gensim LDA
Lda= gensim.models.LdaModel
ldamodel = Lda(DT,num_topics=3,id2word=dictionary,passes=50)
#输出主题结果
print(ldamodel.print_topics(num_topics=3,num_words=3))


