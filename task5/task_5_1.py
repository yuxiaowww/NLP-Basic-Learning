# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:47:12 2019

@author: yuwei
"""

'https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000867&idx=3&sn=e51617747de265f9f242e602d9fed696'

'''
利用tensorflow等工具定义简单的几层网络（激活函数sigmoid）
递归使用链式法则来实现反向传播。
'''

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt 

'1-构造添加一个神经层的函数'
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

'2-导入数据'
#特征一维
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
#为标签加入噪声，更加真实
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
#利用占位符定义我们所需的神经网络的输入
'''
tf.placeholder()就是代表占位符，
这里的None代表无论输入有多少都可以，
因为输入只有一个特征，所以这里是1。
'''
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


'3-搭建网络'
#定义隐藏层，激励函数tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#定义输出层，输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层
prediction = add_layer(l1, 10, 1, activation_function=None)
#计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

'4-梯度下降'
#tf.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss。
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#对变量进行初始化
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  
#定义Session，并用 Session 来执行 init 初始化步骤。 
#（注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）
sess = tf.Session()
sess.run(init)


'5-模型训练'
#机器学习的内容是train_step, 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。
for i in range(1000):
    # 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # 观察迭代改进
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})   

'6-结果可视化'

# 真实值结果可视化
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()

# 预测值结果可视化
fig = plt.figure()
ax = fig.add_subplot(2,1,2)
ax.scatter(x_data, prediction_value)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()


