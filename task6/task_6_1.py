# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:04:20 2019

@author: yuwei
"""

'word2vec调用'

import pandas as pd
from gensim.models import Word2Vec
 
text = [['3', '3', '7', '6', '5', '7', '8', '1'],
        ['2', '2', '3', '9', '1', '8', '3', '7'],
        ['1', '4', '2', '4', '6', '5', '9', '8', '3'],
        ['5', '7', '9', '7', '5', '6', '3', '2', '6'],
        ['8', '2', '5', '8', '8', '1', '7', '8', '3'],
        ['1', '4', '1', '7', '2', '4', '6', '8', '4', '7'],
        ['4', '8', '6', '3', '3', '3', '4', '1', '9', '4'],
        ['7', '4', '2', '5', '1', '6', '4', '3', '1', '4']]
#调用word2vec
model = Word2Vec(text, size=5, min_count=1, window=10, iter=10)
# 输出3的词向量
print(model['3'])  
# 输出text第1行词向量均值
print(pd.DataFrame(model[text[0]]).mean())  
