# coding=utf-8
import os

import input_data
import tensorflow as tf

import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = input_data.read_data_sets("MNIST_data", one_hot=True)

# create model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵作为损失函数，避免均方误差损失函数学习速率降低的问题
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 1，横向比较，每一行 ,比较值最大的数组下标，概率最大的那一个是不是相等，0就是每一列
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 所有的0，1，0，1中求平均值，即为准确率
saver = tf.train.Saver(variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

    path = saver.save(sess,
                      os.path.join(os.path.dirname(__file__), 'data', "regression.ckpt"),
                      write_meta_graph=False,
                      write_state=False)
    print('Saved:', path)
