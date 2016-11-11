#!/usr/bin/env/ python2

import tensorflow as tf
import argparse
import sys
import pandas as pd

from rnn_network import SeizureClassifier
from data_loader import SeizureDataset



# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'model_dir',
    '/tmp/seizure_models/',
     'Directory for trained models.')
flags.DEFINE_string(
    'summaries_dir',
    '/tmp/seizureclassifier',
     'Directory for summaries')
flags.DEFINE_string('train_set', 'train_1', 'Name of the training set.')
flags.DEFINE_string('test_set', 'test_1_new', 'Name of the training set.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
flags.DEFINE_integer(
    'input_subsample_rate',
    5000,
     'Subsampling original input.')
flags.DEFINE_integer('batch_size', 50, 'Size of batches of data to train on.')
flags.DEFINE_boolean('report_train', True, 'If true, performs evaluation and '
                     'report the results at each num_steps iteration')
flags.DEFINE_boolean('eval_net', True, 'If true, perform evaluation after '
                     'a network is trained and report the results.')
flags.DEFINE_boolean(
    'eval_rand',
    True,
     'If true, perform evaluation on random set')


def train_and_validate():
    print('Seizure Detection Learning')
    print('---------------------------------------------------------------')

    do_train = False

    ds_seizure = SeizureDataset(FLAGS.input_subsample_rate,
                                FLAGS.train_set,
                                FLAGS.test_set,
                                batch_size=FLAGS.batch_size)

    if do_train:
        X_train, y_train = ds_seizure.load_train_data(FLAGS.train_set)
        print('Data sample size:', X_train[0].shape)
        print('------------------------------------')
        print('Batch size: ', FLAGS.batch_size)
        print('------------------------------------')

        # Note : Make sure dataset is divisible by batch_size
        try:
            assert(len(X_train) % FLAGS.batch_size == 0)

        except:
            print("Make sure dataset size is divisble by batch_size!")
            sys.exit(1)

        with tf.Graph().as_default():
            # create and train the network
            rnn_net = SeizureClassifier(
                input_timestep=FLAGS.input_subsample_rate,
                batch_size=FLAGS.batch_size)
            rnn_net.setup_loss()
            rnn_net.do_train(ds_seizure, X_train, y_train, FLAGS)
            #print('Final evaluation on the training data')
            #print('---------------------------------------------------------------')
            #rnn_net.do_eval(X_train[0:10], y_train[0:10])

        # At this point we are done with the training data
        del X_train
        del y_train

    print('Testing')
    print('---------------------------------------------------------------')
    X_test, ids = ds_seizure.load_test_data(FLAGS.test_set)
    with tf.Graph().as_default():
        # create and train the network
        rnn_net = SeizureClassifier(
            input_timestep=FLAGS.input_subsample_rate,
            batch_size=FLAGS.batch_size)
        rnn_net.setup_loss()
        predictions = rnn_net.predict(X_test, FLAGS)
        # Save the results
        frame = pd.DataFrame(
            {'File': ids,
             'Class': predictions
             })
        cols = frame.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        frame = frame[cols]
        frame['Class'] = frame['Class'].astype(float)
        frame.to_csv(FLAGS.test_set + '_res.csv', index = False)
        print('Saved results in: ', FLAGS.test_set)


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train_and_validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/melissafm/seizure_models/',
     help='Directory for storing data')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/seizureclassifier',
     help='Directory for storing data')
    parser.add_argument('--train_set', type=str, default='train_2',
                        help='Directory for storing data')
    parser.add_argument('--test_set', type=str, default='test_1_dummy',
                        help='Directory for storing data')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of steps to run trainer.')
    parser.add_argument('--input_subsample_rate', type=int, default=100,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of steps to run trainer.')
    parser.add_argument('--eval_net', type=bool, default=True,
                        help='Do evaluation after a few training iterations.')
    parser.add_argument('--eval_rand', type=bool, default=False,
                        help='Select evaluation set randomly.')

    FLAGS = parser.parse_args()
    tf.app.run()
