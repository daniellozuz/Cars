import tensorflow as tf
import os
import yaml
import sys

import mdl.mnist as mnist
from mdl.model import Model


CONFIG = yaml.load(open(os.path.join('config', sys.argv[1]), 'r'))

FLAGS = tf.app.flags.FLAGS


def train():
    model = Model(**CONFIG)

    with tf.Graph().as_default():
        images, val_images, labels, val_labels = mnist.load_train_data(FLAGS.train_data, CONFIG['num_labels'], CONFIG['image_size'])

        x = tf.placeholder(shape=[None, CONFIG['image_size'], CONFIG['image_size'], 1], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, CONFIG['num_labels']], dtype=tf.float32, name='y')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.contrib.framework.get_or_create_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)
            for i in range(FLAGS.num_iter):
                offset = (i * FLAGS.batch_size) % (len(images) - FLAGS.batch_size)
                batch_x, batch_y = images[offset:(offset + FLAGS.batch_size), :], labels[
                                                                                  offset:(offset + FLAGS.batch_size), :]

                _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                writer.add_summary(summary, i)
                #print(i, cur_loss)
                if i % int(CONFIG['num_iter'] / 10) == 0:
                    validation_accuracy = accuracy.eval(feed_dict={x: val_images, y: val_labels, keep_prob: 1.0})
                    print('Iter {} Accuracy: {}'.format(i, validation_accuracy))

                if i == FLAGS.num_iter - 1:
                    saver.save(sess, FLAGS.checkpoint_file_path, global_step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 16, 'size of training batches')
    tf.app.flags.DEFINE_integer('num_iter', CONFIG['num_iter'], 'number of training iterations')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-' + str(CONFIG['num_iter']), 'path to checkpoint file')
    tf.app.flags.DEFINE_string('train_data', 'data/cars_train.csv', 'path to train and test data')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')

    tf.app.run()
