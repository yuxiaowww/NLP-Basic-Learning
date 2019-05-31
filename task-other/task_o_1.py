# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:26:14 2019

@author: yuwei
"""

'Keras实现文本预处理'
'中文参考文档：https://keras-cn.readthedocs.io/en/latest/preprocessing/text/'

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

text1 = "今天 北京 下 暴雨 了"
text2 = "我 今天 打车 回家"
texts = [text1, text2]

#按空格分割语料
print(text_to_word_sequence(text1))
'''
Tokenizer是一个用于向量化文本，
或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
'''
tokenizer = Tokenizer(num_words=10)
#训练
tokenizer.fit_on_texts(texts)
#处理文档的数量
print(tokenizer.document_count) 
#词频字典，按词频从大到小排序
print(tokenizer.word_counts) 
#保存每个word出现的文档的数量
print(tokenizer.word_docs) 
#给每个词唯一id
print(tokenizer.word_index) 
#保存word的id出现的文档的数量
print(tokenizer.index_docs) 


'https://keras-cn.readthedocs.io/en/latest/preprocessing/sequence/'
# 将序列填充到maxlen长度
print(pad_sequences([[1,2,3],[4,5,6]],maxlen=10,padding='pre')) # 在序列前填充
'''
[[0 0 0 0 0 0 0 1 2 3]
 [0 0 0 0 0 0 0 4 5 6]]
'''
print(pad_sequences([[1,2,3],[4,5,6]],maxlen=10,padding='post')) # 在序列后填充
'''
[[1 2 3 0 0 0 0 0 0 0]
 [4 5 6 0 0 0 0 0 0 0]]
'''


