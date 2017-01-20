# coding=utf-8
"""构建2层全连接网络"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def inference(images, num_image_pixels, num_hidden_units, num_classes, reg_constant=0):
    """构建神经网络, 并进行前向传播计算

    Args:
        images:             图片集
        num_image_pixels:   图片的像素数.
        num_hidden_units:   第一层(隐藏层)的神经元个数.
        num_classes:        图片类别数
        reg_constant:       正则化常数(默认0).

    Returns:
        logits: 计算图片类别的tensor.
    """

    # 第1层
    with tf.variable_scope('Layer1'):
        # 第1层权值
        weights = tf.get_variable(
            name='weights',
            shape=[num_image_pixels, num_hidden_units],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(num_image_pixels))),
            regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
        )

        biases = tf.Variable(tf.zeros([num_hidden_units]), name='biases')

        # 第1层的输出
        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

    # 第2层
    with tf.variable_scope('Layer2'):
        # 第2层权值
        weights = tf.get_variable('weights', [num_hidden_units, num_classes],
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=1.0 / np.sqrt(float(num_hidden_units))),
                                  regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

        biases = tf.Variable(tf.zeros([num_classes]), name='biases')

        # 第2层的输出
        logits = tf.matmul(hidden, weights) + biases

        tf.summary.histogram('logits', logits)

    return logits


def loss(logits, labels):
    """计算logits和labels之间的损失.

  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    loss: Loss tensor, float.
  """

    with tf.name_scope('Loss'):
        # 计算logits和labels间的交叉熵
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, labels, name='cross_entropy'))

        # 最终损失要加上正则化项
        loss = cross_entropy + tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))

        tf.summary.scalar('loss', loss)

    return loss


def training(loss, learning_rate):
    """训练.

    Args:
      loss: Loss tensor
      learning_rate: 学习速率

    Returns:
      train_step: train operation.
    """

    # 创建变量跟踪全局迭代次数
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 梯度下降优化（自动更新迭代次数）
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step


def evaluation(logits, labels):
    """计算准确率.

    Args:
      logits: Logits tensor, float - [batch size, number of classes].
      labels: Labels tensor, int64 - [batch size].

    Returns:
      accuracy: 图片分类准确率
  """

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('train_accuracy', accuracy)

    return accuracy
