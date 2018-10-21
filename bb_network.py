# coding=utf-8

import time
from BBDATA import *
import tensorflow.python as tf
import numpy as np

BBDATA = read_datas('data/')

x_data = tf.placeholder(tf.float32, [None, 144])
y_data = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.truncated_normal([144, 1], stddev=0.2))
b = tf.Variable(tf.constant(0.01, shape=[1]))
y = tf.nn.relu(tf.matmul(x_data, W) + b)

# 按照交叉熵公式计算交叉熵
# cross_entropy = -tf.reduce_mean(y_data * tf.log(y))
cross_entropy = tf.reduce_mean(tf.square((y - y_data)))

# 使用梯度下降法不断的调整变量，寻求最小的交叉熵
# 此处使用梯度下降法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# correct_prediction = tf.equal(y, y_data)
correct_prediction = tf.less_equal(tf.abs(y - y_data), 0.100)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建初始化变量op
init = tf.initialize_all_variables()

# 在session中启动初始化op，以初始化变量
with tf.Session() as sess:
    sess.run(init)
    train_times = 100000
    for i in range(train_times):
        if i % 100 == 0:
            accuracy_value, loss = sess.run([accuracy, cross_entropy],
                                            feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label})
            print "训练次数：" + str(i) + ", 准确率：" + str(accuracy_value * 100) + "%, loss = " + str(loss)

        batch = BBDATA.train_data.batch(100)
        sess.run(train_step, feed_dict={x_data: batch.data, y_data: batch.label})

    # 启动检测函数，输入测试数据集
    accuracy_value = sess.run(accuracy, feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label})
    print "准确率：" + str(accuracy_value * 100) + "%"
    test_data = BBDATA.test_data.batch(1)
    print(test_data.label.head())
    print test_data.label.values
    r = sess.run(y, feed_dict={x_data: test_data.data})
    print "期望结果：" + str(test_data.label.values) + ", 预测结果: " + str(r[0]) + ", 差值 = " + str(r[0] - test_data.label.values)