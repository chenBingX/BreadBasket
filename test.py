# coding=utf-8

import datetime
import numpy as np
from cnn_utils import *

# str = '2016-10-30'
# date = datetime.datetime.strptime(str, '%Y-%m-%d')
# print(date)
# print()

a = weight_variable([2, 135])
a = tf.pad(a, [[0,0],[0,1]], "CONSTANT");
b = tf.reshape(a, [-1, 17, 8, 1])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print(sess.run()))
    print sess.run(b)