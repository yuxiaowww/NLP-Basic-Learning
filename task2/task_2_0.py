# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:18:03 2019

@author: yuwei
"""

#分词基础、词频统计

import pandas as pd
import jieba

#%%
'jieba分词'
seg_list = jieba.cut('我来到北京清华大学',cut_all=True)
print('【全模式】：'+'/'.join(seg_list))

seg_list = jieba.cut('我来到北京清华大学',cut_all=False)
print('【精确模式】：'+'/'.join(seg_list))

seg_list = jieba.cut('我来到北京清华大学')
print('【默认精确模式】：'+', '.join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print('【搜索引擎模式】：'+', '.join(seg_list))

print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
print(jieba.suggest_freq(('中', '将'), True))
print(jieba.suggest_freq(('中将'), True))



#%%
'词性标注'
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))


#%%
'并行分词'

#默认模式
result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

#搜索模式
result = jieba.tokenize(u'永和服装饰品有限公司',mode='search')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

#%%
'常见问题'

#“台中”被切成“台 中”
"解决：强制调高词频"
jieba.add_word('台中')
#或者
jieba.suggest_freq('台中', True)
seg_list = jieba.cut('台中正确应该不会被切开',cut_all=False)
print('【全模式】：'+'/'.join(seg_list))

#“今天天气 不错”应该被切成“今天 天气 不错”
"解决方法：强制调低词频"
jieba.suggest_freq(('今天', '天气'), True)
#或者
jieba.del_word('今天天气')
seg_list = jieba.cut('今天天气不错',cut_all=False)

#切出了词典中没有的词语，效果不理想
"解决方法：关闭新词发现"
seg_list = jieba.cut('丰田太省了')
print('NO HMM:'+'/'.join(seg_list))

seg_list = jieba.cut('丰田太省了', HMM=False)
print('HMM:'+'/'.join(seg_list))


#%%
'词、字符频率统计'

from collections import Counter
data = '北京大学和清华大学是中国的顶尖大学'

#分词后的词频统计
words = list(jieba.cut(data))
print(Counter(words))

#未分词的字符统计
print(Counter(list(data)))







