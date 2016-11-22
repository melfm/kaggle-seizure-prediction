#!/usr/bin/env/ python2
import argparse
import tensorflow as tf
import sys
from cnn_network import SeizureClassifier
from data_loader import SeizureDataset

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'model_dir',
    '/tmp/seizure_models/',
     'Directory for trained models.')

flags.DEFINE_string('train_set', 'train_1', 'Name of the training set.')
flags.DEFINE_string('test_set', 'test_1_new', 'Name of the training set.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')

flags.DEFINE_integer('batch_size', 50, 'Size of batches of data to train on.')
flags.DEFINE_integer('pos_weight', 2, 'Weighted cross entropy const.')
flags.DEFINE_integer('input_dim', 300, 'Size of signal timestep.')


def train_and_validate():
    print('Seizure Detection Learning')
    print('---------------------------------------------------------------')

    do_train = True

    ds_seizure = SeizureDataset(FLAGS)

    if do_train:

        X_train, y_train = ds_seizure.load_train_data(FLAGS.train_set)
        print('Data sample size:', X_train[0].shape)
        print('Length of X_train', len(X_train))
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
            cnn_net = SeizureClassifier(FLAGS)
            cnn_net.setup_loss_and_trainOp(FLAGS)

            X_train_set = X_train
            y_train_set = y_train

            X_test_set = X_train
            y_test_set = y_train

            cnn_net.do_train(ds_seizure,
                             X_train_set,
                             y_train_set,
                             X_test_set,
                             y_test_set,
                             FLAGS)

    # Start a new graph
    with tf.Graph().as_default():
        cnn_net = SeizureClassifier(FLAGS)
        cnn_net.setup_loss_and_trainOp(FLAGS)
        cnn_net.predict(ds_seizure, FLAGS)


def main(_):
    train_and_validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/seizure_models/',
        help='Directory for storing data')

    parser.add_argument(
        '--train_set',
        type=str,
        default='image_train_300_dummy',
     help='Directory for storing data')

    parser.add_argument('--test_set', type=str, default='image_test_300_dummy',
                        help='Directory for storing data')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of steps to run trainer.')

    parser.add_argument('--input_dim', type=int, default=300,
                        help='Subsampling rate.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of steps to run trainer.')

    parser.add_argument('--pos_weight', type=int, default=2,
                        help='Weighted cross entropy const.')

    FLAGS = parser.parse_args()
    tf.app.run()
