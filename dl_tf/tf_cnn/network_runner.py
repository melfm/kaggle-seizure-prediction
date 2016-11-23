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
        train_set_size = int(round(FLAGS.train_ds_ratio * len(X_train)))
        X_train_set = X_train[:train_set_size]
        y_train_set = y_train[:train_set_size]

        X_val_set = X_train[train_set_size:]
        y_val_set = y_train[train_set_size:]
        print('Data sample size = ', X_train[0].shape)
        print('Trainig samples count = ', len(y_train_set))
        print('Validation samples count = ', len(y_val_set))
        print('------------------------------------')
        print('Batch size: ', FLAGS.batch_size)
        print('------------------------------------')

        # Note : Make sure dataset is divisible by batch_size
        # try:
        #     assert(len(X_train) % FLAGS.batch_size == 0)
        # except:
        #     print("Make sure dataset size is divisble by batch_size!")
        #     sys.exit(1)

        with tf.Graph().as_default():
            # create and train the network
            cnn_net = SeizureClassifier(FLAGS)
            cnn_net.setup_loss_and_trainOp(FLAGS)

            cnn_net.do_train(ds_seizure,
                             X_train_set,
                             y_train_set,
                             X_val_set,
                             y_val_set,
                             FLAGS)

    # Start a new graph
    with tf.Graph().as_default():
        cnn_net = SeizureClassifier(FLAGS)
        cnn_net.setup_loss_and_trainOp(FLAGS)
        cnn_net.predict(ds_seizure, FLAGS)


def main(_):
    train_and_validate()


if __name__ == '__main__':

    patient_id = 2

    parser = argparse.ArgumentParser()

    parser.add_argument('--patient_id', type=str, default=patient_id,
                        help='Patient ID, can take1, 2 or 3')

    parser.add_argument('--model_dir', type=str,
                        default='/home/n2mohaje/seizure_models/rings/patient_{0}/'.format(
                            patient_id),
                        help='Directory for storing data')
    parser.add_argument('--train_set', type=str, default='image_train_{0}_300/rings/'.format(
                            patient_id),
                        help='Directory for storing data')

    parser.add_argument('--test_set', type=str, default='image_test_{0}_300/rings/'.format(
                            patient_id),
                        help='Directory for storing data')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of steps to run trainer.')

    parser.add_argument('--input_dim', type=int, default=300,
                        help='Subsampling rate.')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of steps to run trainer.')

    parser.add_argument('--pos_weight', type=int, default=5,
                        help='Weighted cross entropy const.')

    parser.add_argument('--train_ds_ratio', type=int, default=0.75,
                        help='Weighted cross entropy const.')
    FLAGS = parser.parse_args()
    tf.app.run()
