# coding=utf-8

from BBDATA import *
import tensorflow as tf
from cnn_utils import save_model

train_times = 130000
base_path = "/Users/coorchice/Desktop/ML/model/ml/BreadBasket/"
save_path = base_path + str(train_times) + "/"

BBDATA = read_datas('data/')

x_data = tf.placeholder(tf.float32, [None, 135])
y_data = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.truncated_normal([135, 1], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[1]))
y = tf.nn.relu(tf.matmul(x_data, W) + b)

# 按照交叉熵公式计算交叉熵
with tf.name_scope('loss'):
    # cross_entropy = -tf.reduce_sum(y_data * tf.log(y))
    cross_entropy = tf.reduce_mean((tf.square((y - y_data))))
tf.summary.scalar('loss', cross_entropy)

init_lr = 0.00001
global_step = tf.Variable(0., trainable=False)
lr = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps=10000, decay_rate=0.5, staircase=True)

# 使用梯度下降法不断的调整变量，寻求最小的交叉熵
# 此处使用梯度下降法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)

# correct_prediction = tf.equal(y, y_data)
# correct_prediction = tf.less_equal(tf.abs(y - y_data), 150)
# dv = tf.reduce_mean(tf.reduce_sum(tf.abs(y - y_data)))
with tf.name_scope('dv'):
    dv = tf.reduce_mean(tf.abs(y - y_data))
tf.summary.scalar('dv', dv)

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建初始化变量op
init = tf.initialize_all_variables()

add_global = global_step.assign_add(1)
# 在session中启动初始化op，以初始化变量
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('graph', sess.graph)

    sess.run(init)
    global loss2
    loss2 = 0
    for i in range(train_times):
        batch = BBDATA.train_data.batch(200)
        if i % 100 == 0:
            dv_value, loss = sess.run([dv, cross_entropy],
                                      feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label})
            print "训练次数：" + str(i) + ", 差值：" + str(dv_value) + ", train_loss = " + str(
                loss2) + ', test_loss = ' + str(loss)
        summary, train, loss2, _ = sess.run([merged, train_step, cross_entropy, add_global],
                                            feed_dict={x_data: batch.data, y_data: batch.label})
        train_writer.add_summary(summary, i)
    # 启动检测函数，输入测试数据集
    dv_value = sess.run(dv, feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label})
    print "差值：" + str(dv_value) + ""
    test_data = BBDATA.test_data.batch(1)
    print(test_data.label.head())
    print test_data.label.values
    r = sess.run(y, feed_dict={x_data: test_data.data})
    print "期望结果：" + str(test_data.label.values) + ", 预测结果: " + str(r[0]) + ", 差值 = " + str(
        r[0] - test_data.label.values)
    train_writer.close()
    save_model(save_path, sess, train_times)
