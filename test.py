# coding=utf-8

import datetime
import numpy as np
import tensorflow.python as tf

# str = '2016-10-30'
# date = datetime.datetime.strptime(str, '%Y-%m-%d')
# print(date)
# print()

# a = weight_variable([2, 135])
# a = tf.pad(a, [[0,0],[0,1]], "CONSTANT");
# b = tf.reshape(a, [-1, 17, 8, 1])
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     # print(sess.run()))
#     print sess.run(b)

# 新建一个 graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#   c = tf.matmul(a, b)
# # 新建 session with log_device_placement 并设置为 True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 运行这个 op.
# print sess.run(c)

# def get_available_gpus():
#     """
#     code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
#     """
#     from tensorflow.python.client import device_lib as _device_lib
#     local_device_protos = _device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# print(get_available_gpus())

import time

print(time.time() * 1000)