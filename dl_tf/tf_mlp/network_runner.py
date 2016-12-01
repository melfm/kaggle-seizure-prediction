#!/usr/bin/env python
"""Simple neural network architecture for Melbourne iEEG data.
This should achieve an AUC of ~ 0.7%.
"""
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import os
import pdb

from data_loader import SeizureDataset
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
from classifier_ffnn import Classifier_Network
import matplotlib.pyplot as plt

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

class MyDataset:
    def __init__(self,
                 x_train,
                 y_train,
                 x_test = None,
                 y_test = None,
                 train_val_ratio = 0.85):
        x_train = np.reshape(x_train,(len(x_train),-1))
        y_train = np.reshape(y_train,(len(y_train),-1))
        if not x_test == None:
            x_test = np.reshape(x_test,(len(x_test),-1))
        if not y_test == None:
            y_test = np.reshape(y_test,(len(y_test),-1))

        train_set_size = int(round(train_val_ratio * len(x_train)))
        x_train_set = x_train[:train_set_size]
        y_train_set = y_train[:train_set_size]

        x_val_set = x_train[train_set_size:]
        y_val_set = y_train[train_set_size:]

        self.train = DataSet(x_train_set,
                             y_train_set,
                             dtype=dtypes.float32,
                             reshape=False)
        self.validation = DataSet(x_val_set,
                                  y_val_set,
                                  dtype=dtypes.float32,
                                  reshape=False)
        if not (x_test is None) and (y_test is not None):
            self.test = DataSet(x_test,
                                y_test,
                                dtype=dtypes.float32,
                                reshape=False)

def train_and_validate(ds, instance):

    height =1000
    width = 16
    hidden_layer_cnt = 3
    hidden_sizes = np.array([4000, 1000, 50])
    batch_size = FLAGS.batch_size
    hidden_act = tf.nn.relu
    output_act = tf.nn.sigmoid
    keep_prob = .5
    pos_weight = 1.

    print('Seizure Detection Learning')
    print('---------------------------------------------------------------')

    do_train = True


    if do_train:
        print('Data sample size = ', ds.train.num_examples + ds.train.num_examples)
        print('Trainig samples count = ', ds.train.num_examples)
        print('Validation samples count = ', ds.validation.num_examples)
        print('------------------------------------')
        print('Batch size: ', FLAGS.batch_size)
        print('------------------------------------')
        print('Number of ones in the validation set = ',np.sum(ds.validation.labels))

        y_1s_train_cnt = np.sum(ds.train.labels)
        y_0s_train_cnt = ds.train.num_examples - y_1s_train_cnt
        pos_weight =  (1. * y_0s_train_cnt) / y_1s_train_cnt
        print('Positive weight = ', pos_weight)
        with tf.Graph().as_default():
            # create and train the network
            mlp_net = Classifier_Network(FLAGS,
                                         height * width,
                                         1,
                                         hidden_layer_cnt,
                                         hidden_sizes,
                                         batch_size,
                                         hidden_act,
                                         output_act,
                                         keep_prob,
                                         pos_weight)

            mlp_net.setupLoss()

            costs =  mlp_net.train(ds, FLAGS)
            plt.plot(costs)
            plt.savefig(str(FLAGS.patient_id) + '_' + str(instance) + '_res')
            plt.close()

    # Start a new graph
    with tf.Graph().as_default():
        mlp_net = Classifier_Network(FLAGS,
                                     height * width,
                                     1,
                                     hidden_layer_cnt,
                                     hidden_sizes,
                                     batch_size,
                                     hidden_act,
                                     output_act,
                                     keep_prob,
                                     pos_weight)

        mlp_net.setupLoss()
        mlp_net.load()

        predictions = mlp_net.producePredictions(ds.test)
        frame = pd.DataFrame({'File': ds.test.labels.tolist(),
                              'Class': predictions
                              })
        cols = frame.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        frame = frame[cols]
        frame['Class'] = frame['Class'].astype(float)
        frame.to_csv(FLAGS.model_dir + str(FLAGS.patient_id) + '_' + str(instance) + '_res.csv', index=False)
        print('Saved results in: ', FLAGS.test_set)
        return predictions

def main(_):
    username = os.getlogin()
    user_dir = '/home/' + username
    instances_count = 10
<<<<<<< HEAD
    for patient in xrange(3):
        patient_id = 1 + patient
        FLAGS.patient_id = patient_id
        FLAGS.train_set = 'image_train_{0}_1000/single_side_fft/'.format(
                                patient_id)
        FLAGS.test_set = 'image_test_{0}_1000/single_side_fft/'.format(
                                patient_id)
        predictions = 0
        for instances in xrange(instances_count):
            ds_seizure = SeizureDataset(FLAGS)
            X_train, y_train = ds_seizure.load_train_data(FLAGS.train_set)
            X_test, y_ids = ds_seizure.load_test_data(FLAGS.test_set)
            ds = MyDataset(X_train,y_train,X_test,y_ids)
            FLAGS.model_dir='/home/n2mohaje/seizure_models/single_side_fft/patient_{0}/model_{1}.cpkd'.format(
                FLAGS.patient_id,instances)
            predictions += np.array(train_and_validate(ds,instances))

        frame = pd.DataFrame({'File': ds.test.labels.tolist(),
                                'Class': list(predictions / instances_count)
                                })
        cols = frame.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        frame = frame[cols]
        frame['Class'] = frame['Class'].astype(float)
        list2d = frame['File'].values.tolist()
        merged = list(itertools.chain.from_iterable(list2d))
        frame['File'] = merged
        frame.to_csv(str(FLAGS.patient_id) + '_average_res.csv', index=False)
        print('Saved results in: ', FLAGS.test_set)
=======
    # for patient in xrange(3):
    patient_id = 0 #1 + patient
    FLAGS.patient_id = patient_id

    FLAGS.train_set = 'image_train_all_1000/single_side_fft/'.format(
                            patient_id)
    FLAGS.test_set = 'image_test_all_1000/single_side_fft/'.format(
                            patient_id)
    predictions = 0
    for instances in xrange(instances_count):
        ds_seizure = SeizureDataset(FLAGS)
        X_test, y_ids = ds_seizure.load_test_data(FLAGS.test_set)
        X_train, y_train = ds_seizure.load_train_data(FLAGS.train_set)
        ds = MyDataset(X_train,y_train,X_test,y_ids)
        FLAGS.model_dir= user_dir +'/seizure_models/single_side_fft/patient_{0}/model_{1}.cpkd'.format(
            FLAGS.patient_id,instances)
        predictions += np.array(train_and_validate(ds,instances))

    frame = pd.DataFrame({'File': ds.test.labels.tolist(),
                            'Class': list(predictions / instances_count)
                            })
    cols = frame.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    frame = frame[cols]
    frame['Class'] = frame['Class'].astype(float)
    list2d = frame['File'].values.tolist()
    merged = list(itertools.chain.from_iterable(list2d))
    frame['File'] = merged
    frame.to_csv(str(FLAGS.patient_id) + '_average_res.csv', index=False)
    print('Saved results in: ', FLAGS.test_set)
>>>>>>> 682a8d38127b985f542f555ef926af18301f8946

if __name__ == '__main__':

        patient_id = 0
        username = os.getlogin()
        user_dir = '/home/' + username

        parser = argparse.ArgumentParser()

        parser.add_argument('--patient_id', type=int, default=patient_id,
                            help='Patient ID, can take 1, 2 or 3')

        parser.add_argument('--model_dir', type=str,
                            default= user_dir + '/seizure_models/single_side_fft/patient_{0}/'.format(
                                patient_id),
                            help='Directory for storing data')
        #parser.add_argument('--train_set', type=str, default='image_train_{0}_1000/single_side_fft/'.format(
        parser.add_argument('--train_set', type=str, default='train_{0}_dummy/'.format(
                                patient_id),
                            help='Directory for storing data')

        #parser.add_argument('--test_set', type=str, default='image_test_{0}_1000/single_side_fft/'.format(
        parser.add_argument('--test_set', type=str, default='test_{0}_dummy/'.format(
                                patient_id),
                            help='Directory for storing data')

        parser.add_argument('--learning_rate', type=float, default=0.0005,
                            help='Initial learning rate')

        parser.add_argument('--epochs', type=int, default=4000,
                            help='Number of steps to run trainer.')

        parser.add_argument('--batch_size', type=int, default=40,
                            help='Number of steps to run trainer.')

        parser.add_argument('--pos_weight', type=float, default=10.,
                            help='Weighted cross entropy const.')

        parser.add_argument('--train_ds_ratio', type=float, default=0.85,
                            help='Weighted cross entropy const.')

        parser.add_argument('--save', type=bool, default=True,
                            help='Set to True to save the best model.')

        parser.add_argument('--max_localmin_iters', type=int, default=100,
                            help='Maximum number of iterations allowed to be spent on a local minimum.')

        parser.add_argument('--max_perturbs', type=int, default=3,
                            help='Maximum number of weight perturbations.')
        FLAGS = parser.parse_args()
        tf.app.run()






