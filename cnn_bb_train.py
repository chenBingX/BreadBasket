# coding=utf-8

import time
from cnn_model import *
from BBDATA import *

train_times = 10000
base_path = "/Users/coorchice/Desktop/ML/model/cnn/BreadBasket/"
save_path = base_path + str(train_times) + "/"

# 读取数据
BBDATA = read_datas('data/')

# 创建网络
network = CnnBreadBasketNetwork()
x_data = network.x_data
y_data = network.y_data
y_conv = network.y_conv
keep_prob = network.keep_prob
# ------------------------构建损失函数---------------------
with tf.name_scope("cross_entropy"):
    # 创建正则化对象，此处使用的是 L2 范数
    regularization = tf.contrib.layers.l2_regularizer(scale=(5.0 / 50000))
    # 应用正则化到参数集上
    reg_term = tf.contrib.layers.apply_regularization(regularization)
    # 在损失函数中加入正则化项
    # cross_entropy = (-tf.reduce_sum(y_data * tf.log(y_conv)) + reg_term)
    cross_entropy = tf.reduce_mean(tf.square((y_conv - y_data))) + reg_term
    tf.summary.scalar('loss', cross_entropy)
with tf.name_scope("train_step"):
    # 使用 Adam 进行损失函数的梯度下降求解
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("difference_value"):
    dv = tf.reduce_mean(tf.abs(y_conv - y_data))
    tf.summary.scalar('difference_value', cross_entropy)

# ------------------------构建模型评估函数---------------------
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.less_equal(tf.abs(y_conv - y_data), 0.2)
    with tf.name_scope("accuracy"):
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# 创建会话
sess = tf.InteractiveSession()

summary_merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_path + "graph/train", sess.graph)
test_writer = tf.summary.FileWriter(save_path + "graph/test")

start_time = int(round(time.time() * 1000))

# 初始化参数
sess.run(tf.initialize_all_variables())

global loss
loss = 0
global train_accuracy
for i in range(train_times):
    # 从训练集中取出 100 个样本进行一波训练
    batch = BBDATA.train_data.batch(50)

    if i % 100 == 0:
        summary, train_accuracy, test_loss, dv_value = sess.run([summary_merged, accuracy, cross_entropy, dv],
                                           feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label, keep_prob: 1})
        test_writer.add_summary(summary, i)
        consume_time = int(round(time.time() * 1000)) - start_time
        print("当前共训练 " + str(i) + "次, 累计耗时：" + str(consume_time) + "ms，实时准确率为：%g" % (train_accuracy * 100.) + "%, "
             + "当前误差均值：" + str(dv_value) + ", train_loss = " + str(loss) + ", test_loss = " + str(test_loss))
    if i % 1000 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _, loss = sess.run([summary_merged, train_step, cross_entropy],
                              feed_dict={x_data: batch.data, y_data: batch.label, keep_prob: 0.5}, options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, str(i))
        train_writer.add_summary(summary, i)
    else:
        summary, _, loss = sess.run([summary_merged, train_step, cross_entropy],
                              feed_dict={x_data: batch.data, y_data: batch.label, keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    # 每训练 5000 次保存一次模型
    if i != 0 and i % 5000 == 0:
        test_accuracy, test_dv = sess.run([accuracy, dv], feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label, keep_prob: 1})
        save_model(base_path + str(i) + "_" + str(test_dv) + "/", sess, i)

# 在测试集计算准确率
summary, test_accuracy, test_dv = sess.run([summary_merged, accuracy, dv],
                                  feed_dict={x_data: BBDATA.test_data.data, y_data: BBDATA.test_data.label, keep_prob: 1})
train_writer.add_summary(summary)
print("测试集准确率：%g" % (test_accuracy) + ", 误差均值：" + str(test_dv))

print("训练完成！")
train_writer.close()
test_writer.close()
# 保存模型
save_model(save_path, sess, train_times)