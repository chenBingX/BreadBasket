# coding=utf-8

import time
from data_utils import *

x_data = tf.placeholder(tf.float32, [None, 3])
y_data = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.zeros([3, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.softmax(tf.matmul(x_data, W) + b)

# 按照交叉熵公式计算交叉熵
cross_entropy = -tf.reduce_sum(y_data * tf.log(y))

# 使用梯度下降法不断的调整变量，寻求最小的交叉熵
# 此处使用梯度下降法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(y, train_y.values)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建初始化变量op
init = tf.initialize_all_variables()

# 在session中启动初始化op，以初始化变量
with tf.Session() as sess:
    sess.run(init)
    train_times = 1000
    for i in range(train_times):
        sess.run(train_step, feed_dict={x_data: train_data.values, y_data: train_y.values})

# 启动检测函数，输入测试数据集
accuracy_value = sess.run(accuracy, feed_dict={x_data: test_data.values, y_data: test_y.values})
print "准确率：" + str(accuracy_value * 100) + "%"
