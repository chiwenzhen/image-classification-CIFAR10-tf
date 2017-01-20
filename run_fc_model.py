# coding=utf-8
"""训练并评估神经网络准确率-CIFAR-10"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import two_layer_fc

# 神经网络模型参数
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
                     'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
                    'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

# 日志放在不同文件夹
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# np.random.seed(1)
# tf.set_random_seed(1)

# 加载CIFAR-10数据
data_sets = data_helpers.load_data()

# -----------------------------------------------------------------------------
# 定义tensorflow流图
# -----------------------------------------------------------------------------

# 定义placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# 预测：前向传播
logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
                                FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# 计算预测值的损失
loss = two_layer_fc.loss(logits, labels_placeholder)

# 训练：后向传播
train_step = two_layer_fc.training(loss, FLAGS.learning_rate)

# 计算预测值的准确率
accuracy = two_layer_fc.evaluation(logits, labels_placeholder)

# 准备TensorBoard的日志数据
summary = tf.summary.merge_all()

# 用于在指定时机保存模型
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# 运行tensorflow流图
# -----------------------------------------------------------------------------

with tf.Session() as sess:
    # 初始化变量，并创建summary-writer
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    # 产生批量训练数据
    zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
    batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
                                     FLAGS.max_steps)

    for i in range(FLAGS.max_steps):

        # 获取下一批训练数据
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
            images_placeholder: images_batch,
            labels_placeholder: labels_batch
        }

        # 打印当前准确率
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)

        # 一次训练
        sess.run([train_step, loss], feed_dict=feed_dict)

        # 每1000次迭代保存一次模型
        if (i + 1) % 1000 == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            print('Saved checkpoint')

    # 测试集的准确率
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
