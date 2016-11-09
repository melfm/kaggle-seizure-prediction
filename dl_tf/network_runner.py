#!/usr/bin/env/ python2

import tensorflow as tf
import argparse

from rnn_network import SeizureClassifier as net
from data_loader import SeizureDataset as dl


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
flags.DEFINE_boolean('report_train', True, 'If true, performs evaluation and '
                                          'report the results at each num_steps iteration')
flags.DEFINE_boolean('report_net', True, 'If true, perform evaluation after '
                                          'a network is trained and report the results.')

def train_and_validate():
    print('Seizure Detection Learning')
    print('---------------------------------------------------------------')






def main(_):
    train_and_validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of steps to run trainer.')
    FLAGS = parser.parse_args()
    tf.app.run()
