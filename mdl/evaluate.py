import tensorflow as tf
import numpy as np
import os
import yaml
import sys
import datetime

from mdl import data_loader
from mdl.model import Model


CONFIG = yaml.load(open(os.path.join('config', sys.argv[1]), 'r'))

FLAGS = tf.app.flags.FLAGS


def evaluate():
    with tf.Graph().as_default():
        images, labels = data_loader.load_test_data(FLAGS.test_data, **CONFIG)
        model = Model(**CONFIG)
        logits = model.inference(images, keep_prob=1.0)
        accuracy = model.accuracy(logits, labels)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_file_path)

            confusion_matrix = tf.confusion_matrix(tf.argmax(logits, 1), tf.argmax(labels, 1))
            confusion_matrix = sess.run([confusion_matrix])
            print(confusion_matrix)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            total_accuracy = sess.run([accuracy])
            print('Test accuracy: {}'.format(total_accuracy))
            better = 100 * (total_accuracy[0] * CONFIG['num_labels'] - 1)
            print('{}% better than random'.format(better))
            with open('results.txt', 'a') as results_file: # TODO make result file per yml config
                results_file.write('Date: {} Filename: {} Accuracy: {:.4f} Gain: {:.2f}% {}\n'.format(timestamp, sys.argv[1], total_accuracy[0], better, CONFIG))

            file_prefix = sys.argv[1] + '.' + timestamp
            np.save('results/' + file_prefix + '.cm.npy', confusion_matrix)
            np.save('results/' + file_prefix + '.acc.npy', total_accuracy[0])
            np.save('results/' + file_prefix + '.better.npy', better)
            np.save('results/' + file_prefix + '.cfg.npy', CONFIG)

def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-' + str(CONFIG['num_iter']) + '-' + str(CONFIG['num_iter']), 'path to checkpoint file')
    tf.app.flags.DEFINE_string('test_data', 'data/cars_test.csv', 'path to test data')

    tf.app.run()
