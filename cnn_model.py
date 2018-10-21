# coding=utf-8
from cnn_utils import *


class CnnBreadBasketNetwork:
    def __init__(self):
        with tf.name_scope("input"):
            self.x_data = tf.placeholder(tf.float32, shape=[None, 144], name='x_data')
            with tf.name_scope("input_reshape"):
                input_data = tf.reshape(self.x_data, [-1, 12, 12, 1])
                tf.image_summary("input", input_data, 1)
            self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')

        # ------------------------构建第一层网络---------------------
        with tf.name_scope("hidden1"):
            with tf.name_scope("weights"):
                W_conv1 = weight_variable([3, 3, 1, 32])
                variable_summaries(W_conv1, "W_conv1")
            with tf.name_scope("biases"):
                b_conv1 = bias_variable([32])
                variable_summaries(b_conv1, "b_conv1")
            h_conv1 = tf.nn.relu(conv2(input_data, W_conv1) + b_conv1)
            tf.histogram_summary('activations_h_conv1', h_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            tf.histogram_summary('pools_h_pool1', h_pool1)

        # ------------------------构建第二层网络---------------------
        with tf.name_scope("hidden2"):
            with tf.name_scope("weights"):
                W_conv2 = weight_variable([5, 5, 32, 64])
                variable_summaries(W_conv2, 'W_conv2')
            with tf.name_scope("biases"):
                b_conv2 = bias_variable([64])
                variable_summaries(b_conv2, 'b_conv2')
            h_conv2 = tf.nn.relu(conv2(h_pool1, W_conv2) + b_conv2)
            tf.histogram_summary('activations_h_conv2', h_conv2)
            self.h_pool2 = max_pool_2x2(h_conv2)
            tf.histogram_summary('pools_h_pool2', self.h_pool2)

        # ------------------------构建全链接层---------------------
        with tf.name_scope("fc1"):
            with tf.name_scope("weights"):
                W_fc1 = weight_variable([3 * 3 * 64, 1024])
                variable_summaries(W_fc1, 'W_fc1')
            with tf.name_scope("biases"):
                b_fc1 = bias_variable([1024])
                variable_summaries(b_fc1, 'b_fc1')
            # with tf.name_scope("fc1_reshape"):
            h_pool2_flat = tf.reshape(self.h_pool2, [-1, 3 * 3 * 64])
                # tf.image_summary("fc1", h_pool2_flat, 1)
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            tf.histogram_summary('activations_h_fc1', h_fc1)

        # ------------------------添加Dropout减少过拟合---------------------
        # with tf.name_scope("dropout"):
        #     self.keep_prob = tf.placeholder(tf.float32)
        #     tf.scalar_summary('dropout_keep_probability', self.keep_prob)
        #     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # ------------------------构建输出层---------------------
        with tf.name_scope("output"):
            with tf.name_scope("weights"):
                W_fc2 = weight_variable([1024, 1])
                variable_summaries(W_fc2, 'W_out')
            with tf.name_scope("biases"):
                b_fc2 = bias_variable([1])
                variable_summaries(b_fc2, 'b_out')
            self.y_conv = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
            tf.histogram_summary('activations_y_conv', self.y_conv)

