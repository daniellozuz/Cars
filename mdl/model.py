import tensorflow as tf
import os


class Model(object):
    def __init__(self, num_labels, image_size, k, c1, c2, p, batch_size=16, learning_rate=1e-4, **kwargs):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels
        self._image_size = image_size
        self._k = k
        self._c1 = c1
        self._c2 = c2
        self._p = p

    def inference(self, images, keep_prob):
        with tf.variable_scope('conv1') as scope:
            kernel = self._create_weights([self._k, self._k, 1, self._c1])
            conv = self._create_conv2d(images, kernel)
            bias = self._create_bias([self._c1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv1 = tf.nn.relu(preactivation, name=scope.name)
            self._activation_summary(conv1)

        # pool 1
        h_pool1 = self._create_max_pool_2x2(conv1)

        with tf.variable_scope('conv2') as scope:
            kernel = self._create_weights([self._k, self._k, self._c1, self._c2])
            conv = self._create_conv2d(h_pool1, kernel)
            bias = self._create_bias([self._c2])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(preactivation, name=scope.name)
            self._activation_summary(conv2)

        # pool 2
        h_pool2 = self._create_max_pool_2x2(conv2)

        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(h_pool2, [-1, (self._image_size // 4) * (self._image_size // 4) * self._c2])
            W_fc1 = self._create_weights([(self._image_size // 4) * (self._image_size // 4) * self._c2, self._p])
            b_fc1 = self._create_bias([self._p])
            local1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name=scope.name)
            self._activation_summary(local1)

        with tf.variable_scope('local2_linear') as scope:
            W_fc2 = self._create_weights([self._p, self._num_labels])
            b_fc2 = self._create_bias([self._num_labels])
            local1_drop = tf.nn.dropout(local1, keep_prob)
            local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
            self._activation_summary(local2)
        return local2

    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cost = tf.reduce_mean(cross_entropy, name=scope.name)
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, y):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), dtype=tf.float32),
                                      name=scope.name)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _create_conv2d(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    def _create_max_pool_2x2(self, input):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))