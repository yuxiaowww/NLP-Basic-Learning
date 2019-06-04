# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:37:37 2019

@author: yuwei
"""

'one-hot方法'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder
 
'one-hot方法1：get_dummies'
print(pd.get_dummies(['bb', 'aa', 'cc', 'dd', 'bb', 'dd', 'aa', 'bb']))
 
'one-hot方法2：sklearn导包'
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
data = [[1, 1, 2],
        [2, 2, 1],
        [4, 3, 2],
        [1, 4, 2]]
print(ohe.fit_transform(data))
