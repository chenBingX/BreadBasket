# coding=utf-8
from cnn_utils import *


class CnnBreadBasketNetwork:
    def __init__(self):
        with tf.name_scope("input"):
            self.x_data = tf.placeholder(tf.float32, shape=[None, 135], name='x_data')
            input_data = tf.pad(self.x_data, [[0, 0], [0, 1]], 'CONSTANT')
            with tf.name_scope("input_reshape"):
                input_data = tf.reshape(input_data, [-1, 17, 8, 1])
                tf.summary.image("input", input_data, 1)
            self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')

        # ------------------------构建第一层网络---------------------
        with tf.name_scope("hidden1"):
            # 第一个卷积
            with tf.name_scope("weights1"):
                W_conv11 = weight_variable([3, 3, 1, 64])
                variable_summaries(W_conv11, "W_conv11")
            with tf.name_scope("biases1"):
                b_conv11 = bias_variable([64])
                variable_summaries(b_conv11, "b_conv11")
            h_conv11 = tf.nn.relu(conv2(input_data, W_conv11) + b_conv11)
            tf.summary.histogram('activations_h_conv11', h_conv11)
            # 第二个卷积
            with tf.name_scope("weights2"):
                W_conv12 = weight_variable([3, 3, 64, 64])
                variable_summaries(W_conv12, "W_conv12")
            with tf.name_scope("biases2"):
                b_conv12 = bias_variable([64])
                variable_summaries(b_conv12, "b_conv12")
            h_conv12 = tf.nn.relu(conv2(h_conv11, W_conv12) + b_conv12)
            tf.summary.histogram('activations_h_conv11', h_conv12)
            # 池化
            h_pool1 = max_pool_2x2(h_conv12)
            tf.summary.histogram('pools_h_pool1', h_pool1)

        # ------------------------构建第二层网络---------------------
        with tf.name_scope("hidden2"):
            # 第一层
            with tf.name_scope("weights1"):
                W_conv21 = weight_variable([5, 5, 64, 128])
                variable_summaries(W_conv21, 'W_conv21')
            with tf.name_scope("biases1"):
                b_conv21 = bias_variable([128])
                variable_summaries(b_conv21, 'b_conv21')
            h_conv21 = tf.nn.relu(conv2(h_pool1, W_conv21) + b_conv21)
            tf.summary.histogram('activations_h_conv21', h_conv21)
            # 第二层
            with tf.name_scope("weights2"):
                W_conv22 = weight_variable([5, 5, 128, 128])
                variable_summaries(W_conv22, 'W_conv22')
            with tf.name_scope("biases2"):
                b_conv22 = bias_variable([128])
                variable_summaries(b_conv22, 'b_conv22')
            h_conv22 = tf.nn.relu(conv2(h_conv21, W_conv22) + b_conv22)
            tf.summary.histogram('activations_h_conv22', h_conv22)
            # 池化
            self.h_pool2 = max_pool_2x2(h_conv22)
            tf.summary.histogram('pools_h_pool2', self.h_pool2)

        shape_0 = self.h_pool2.get_shape()[1].value
        print('shape_0 = ' + str(shape_0))
        shape_1 = self.h_pool2.get_shape()[2].value
        print('shape_1 = ' + str(shape_1))
        h_pool2_flat = tf.reshape(self.h_pool2, [-1, shape_0 * shape_1 * 128])

        self.keep_prob = tf.placeholder(tf.float32)
        # ------------------------ 构建全链接层一 ---------------------
        with tf.name_scope("fc1"):
            with tf.name_scope("weights"):
                W_fc1 = weight_variable([shape_0 * shape_1 * 128, 4096])
                variable_summaries(W_fc1, 'W_fc1')
            with tf.name_scope("biases"):
                b_fc1 = bias_variable([4096])
                variable_summaries(b_fc1, 'b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            tf.summary.histogram('activations_h_fc1', h_fc1)
        with tf.name_scope("dropout1"):
            tf.summary.scalar('dropout_keep_probability1', self.keep_prob)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # ------------------------ 构建全链接层二 ---------------------
        with tf.name_scope("fc2"):
            with tf.name_scope("weights"):
                W_fc2 = weight_variable([4096, 4096])
                variable_summaries(W_fc2, 'W_fc2')
            with tf.name_scope("biases"):
                b_fc2 = bias_variable([4096])
                variable_summaries(b_fc2, 'b_fc2')
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            tf.summary.histogram('activations_h_fc2', h_fc2)
        with tf.name_scope("dropout2"):
            tf.summary.scalar('dropout_keep_probability2', self.keep_prob)
            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # ------------------------构建输出层---------------------
        with tf.name_scope("output"):
            with tf.name_scope("weights"):
                W_out = weight_variable([4096, 1])
                variable_summaries(W_out, 'W_out')
            with tf.name_scope("biases"):
                b_out = bias_variable([1])
                variable_summaries(b_out, 'b_out')
            self.y_conv = tf.nn.relu(tf.matmul(h_fc2_drop, W_out) + b_out)
            tf.summary.histogram('activations_y_conv', self.y_conv)
