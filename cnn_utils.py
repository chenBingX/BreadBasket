# coding=utf-8

import tensorflow.python as tf
import os

# 创建权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    # 加入正则化
    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
    return var

# 创建偏置量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 创建一个卷积核
def conv2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

# 创建一个池化核
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def save_model(path, sess, train_times):
    """
    保存模型
    :param path: 保存路径
    :param sess: sess
    :param train_times: 训练次数
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    saver = tf.train.Saver()
    saver.save(sess, path, train_times)
    print("保存模型成功！模型路径：" + path)

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean_' + name, mean)
      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('stddev_' + name, stddev)
      tf.scalar_summary('max_' + name, tf.reduce_max(var))
      tf.scalar_summary('min_' + name, tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.histogram_summary('histogram_' + name, var)