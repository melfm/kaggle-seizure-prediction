#!/usr/
import argparse
import tensorflow as tf
import sys
from cnn_network import SeizureClassifier
from data_loader import SeizureDataset

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('patient_id', 1,'Patient ID, can take 1, 2 or 3')
flags.DEFINE_string('model_dir', '/tmp/seizure_models/', 'Directory for trained models.')

flags.DEFINE_string('train_set', 'train_1', 'Name of the training set.')
flags.DEFINE_string('test_set', 'test_1_new', 'Name of the training set.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')

flags.DEFINE_integer('batch_size', 50, 'Size of batches of data to train on.')
flags.DEFINE_integer('pos_weight', 2, 'Weighted cross entropy const.')
flags.DEFINE_integer('train_ds_ratio', 0.75, 'Weighted cross entropy const.')
flags.DEFINE_integer('save', True, 'Set to True to save the best model.')


def train_and_validate():
    height = 300
    width = 300
    depth = 16
    output_dim = 1
    patch_size_1 = 3
    patch_size_2 = 5
    feature_size_1 = 64
    feature_size_2 = 128
    fc_size = 2048

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
            cnn_net = SeizureClassifier(FLAGS,
                                        height,
                                        width,
                                        depth,
                                        output_dim,
                                        patch_size_1,
                                        patch_size_2,
                                        feature_size_1,
                                        feature_size_2,
                                        fc_size)
            cnn_net.setup_loss_and_trainOp(FLAGS)

            cnn_net.do_train(ds_seizure,
                             X_train_set,
                             y_train_set,
                             X_val_set,
                             y_val_set,
                             FLAGS)

    # Start a new graph
    with tf.Graph().as_default():
        cnn_net = SeizureClassifier(FLAGS,
                                    height,
                                    width,
                                    depth,
                                    output_dim,
                                    patch_size_1,
                                    patch_size_2,
                                    feature_size_1,
                                    feature_size_2,
                                    fc_size)
        cnn_net.setup_loss_and_trainOp(FLAGS)
        cnn_net.predict(ds_seizure, FLAGS)


def main(_):
    train_and_validate()


if __name__ == '__main__':

    patient_id = 1

    parser = argparse.ArgumentParser()

    parser.add_argument('--patient_id', type=int, default=patient_id,
                        help='Patient ID, can take 1, 2 or 3')

    parser.add_argument('--model_dir', type=str,
                        default='/home/n2mohaje/seizure_models/resp_ffts/patient_{0}/'.format(
                            patient_id),
                        help='Directory for storing data')
    parser.add_argument('--train_set', type=str, default='image_train_{0}_300/resp_ffts/'.format(
    # parser.add_argument('--train_set', type=str, default='train_{0}_dummy/'.format(
                            patient_id),
                        help='Directory for storing data')

    parser.add_argument('--test_set', type=str, default='image_test_{0}_300/resp_ffts/'.format(
    # parser.add_argument('--test_set', type=str, default='test_{0}_dummy/'.format(
                            patient_id),
                        help='Directory for storing data')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of steps to run trainer.')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of steps to run trainer.')

    parser.add_argument('--pos_weight', type=float, default=2.,
                        help='Weighted cross entropy const.')

    parser.add_argument('--train_ds_ratio', type=float, default=0.75,
                        help='Weighted cross entropy const.')
    parser.add_argument('--save', type=bool, default=False,
                        help='Set to True to save the best model.')
    FLAGS = parser.parse_args()
    tf.app.run()
