# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:48:53 2019

@author: yuwei
"""

#%%
'TF-IDF'

"方法1：TfidfVectorizer类"
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I come to China to travel",
          "This is a car polupar in China",
          "I love tea and Apple ",
          "The work is to write some papers in science"]

#最大特征、最小词频、最大词频
tfidf_model = TfidfVectorizer(max_features=10,min_df=2,max_df=5).fit_transform(corpus)
#tfidf数组
print(tfidf_model.toarray())
#tfidf矩阵
print(tfidf_model.todense())


"方法2：TfidfTransformer类,仅针对数字"
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
corpus = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]
tfidf_model = transformer.fit_transform(corpus)
print(tfidf_model.toarray())


"方法3：CountVectorizer+TfidfTransformer的组合"
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
corpus=["I come to China to travel",
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"]

#将词语转换成词频矩阵
vectorizer=CountVectorizer()
#将统计每个词语的tf-idf权值
transformer = TfidfTransformer()

tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
print(tfidf.toarray())

"方法4：使用gensim提取文本的tfidf特征"
from gensim import corpora, models
corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document']
#按空格分词
word_list = [ sentence.split(' ') for sentence in corpus]
#赋给语料库中每个词(不重复的词)一个整数id
dictionary = corpora.Dictionary(word_list)
#通过下面的方法可以看到语料库中每个词对应的id
print(dictionary.token2id)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
    
#载入模型
tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")
    
#使用模型计算tfidf值
tfidf = models.TfidfModel.load("my_model.tfidf")
tfidf_vec = []
for text in corpus:
    string_bow = dictionary.doc2bow(text.lower().split())
    tfidf_vec.append(tfidf[string_bow])
print(tfidf_vec)

#%%
'互信息'
from sklearn import datasets
from sklearn import metrics as mr

iris = datasets.load_iris()
#150*4特征
x = iris.data
y = iris.target

#计算互信息
print(mr.mutual_info_score(x[:,0],y))
print(mr.mutual_info_score(x[:,1],y))
print(mr.mutual_info_score(x[:,2],y))
print(mr.mutual_info_score(x[:,3],y))


#%%
'word2vec'

from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

raw_sentences = ["the quick brown fox jumps over the lazy dogs",
                 "yoyoyo you go home now to sleep"]
sentences= [s.split() for s in raw_sentences]
model = Word2Vec(sentences, min_count=1)

# 进行相关性比较
model.similarity('dogs','you')



