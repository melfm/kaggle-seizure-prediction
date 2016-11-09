#!/usr/bin/env/ python2

import tensorflow as tf
import argparse
import sys

from rnn_network import SeizureClassifier
from data_loader import SeizureDataset


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('result_dir', 'result', 'Directory to put the results.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
flags.DEFINE_boolean('input_subsample_rate', 1000, 'Subsampling original input.')
flags.DEFINE_boolean('batch_size', 10, 'Size of batches of data to train on.')
flags.DEFINE_boolean('report_train', True, 'If true, performs evaluation and '
                                          'report the results at each num_steps iteration')
flags.DEFINE_boolean('report_net', True, 'If true, perform evaluation after '
                                          'a network is trained and report the results.')

def train_and_validate():
    print('Seizure Detection Learning')
    print('---------------------------------------------------------------')
    train_set_name = 'train_1_dummy'
    test_set_name = 'test_1_dummy'
    batch_size = 1
    do_train = True

    ds_seizure = SeizureDataset(FLAGS.input_subsample_rate,
                                train_set_name,
                                test_set_name,
                                batch_size)

    if do_train :
        X_train, y_train = ds_seizure.load_train_data(train_set_name)
        # Note : Make sure dataset is divisible by batch_size
        try:
            assert(len(X_train) % batch_size == 0)

        except:
            print("Make sure dataset size is divisble by batch_size!")
            sys.exit(1)

        with tf.Graph().as_default():
            # create and train the network
            rnn_net = SeizureClassifier(batch_size=batch_size)
            rnn_net.setup_loss()
            rnn_net.do_train(ds_seizure, X_train, y_train, FLAGS)






def main(_):
    train_and_validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='/tmp/predictions',
                        help='Directory for storing data')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of steps to run trainer.')
    parser.add_argument('--input_subsample_rate', type=int, default=1000,
                        help='Number of steps to run trainer.')
    FLAGS = parser.parse_args()
    tf.app.run()
