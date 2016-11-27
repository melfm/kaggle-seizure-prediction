#! /usr/bin/env/ python2
from sklearn import preprocessing
from operator import itemgetter
import scipy.io
import numpy as np
import pandas as pd
import random
import os

import pdb
import matplotlib.pyplot as plt


class SeizureDataset:

    #####################
    INTERICTAL_CLASS = 0
    PREICTAL_CLASS = 1
    #####################
    train_class_ratio = 0.24

    path_to_all_datasets = os.path.abspath(os.path.join(
                        '../..', 'data_dir/Kaggle_data/data'))
    safe_label_files = 'train_and_test_data_labels_safe.csv'

    def __init__(self, FLAGS):
        self.train_set = FLAGS.train_set
        self.test_set = FLAGS.test_set
        self.batch_size = FLAGS.batch_size

        self.index_0 = 0
        self.batch_index = self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_index = self.batch_size

    def get_data_dir(self, data_set_name):

        path_to_dataset = os.path.join(
            self.path_to_all_datasets, data_set_name)
        print('Loading data set:\n', path_to_dataset)
        data_files = os.listdir(path_to_dataset)
        for mat_file in data_files:
            assert(mat_file.endswith('.mat')) == True
        return data_files

    def get_class_from_name(self, name):
        """
        Gets the class from the file name.
        The class is defined by the last number written in the file name.
        For example:
        Input: ".../1_1_1.mat"
        Output: 1.0
        Input: ".../1_1_0.mat"
        Output: 0.0
        """
        try:
            return float(name[-5])
        except:
            return 0.0

    def count_class_occurrences(self, data_files_all):
        # Count the occurrences of Interictal and Preictal classes
        interictal_count = 0
        preictal_count = 0
        for data in data_files_all:
            if data['class'] == self.INTERICTAL_CLASS:
                interictal_count += 1
            elif data['class'] == self.PREICTAL_CLASS:
                preictal_count += 1

        return interictal_count, preictal_count

    def get_file_names_and_classes(self, data_dir_name):
        # Check for unsafe files here
        safe_labels = os.path.join(
            self.path_to_all_datasets,
            self.safe_label_files)
        df_safe = pd.read_csv(safe_labels)

        ignored_files = df_safe.loc[df_safe['safe'] == 0]
        ignored_files = ignored_files['image'].tolist()
        ignored_files.append('1_45_1.mat')
        #print('Ignoring these files:', ignored_files)
        all_data = self.get_data_dir(data_dir_name)

        file_with_class = np.array(
            [(mat_file, self.get_class_from_name(mat_file))
             for mat_file in all_data if mat_file not in ignored_files],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])
        return file_with_class

    def pick_random_observation(self, data_dir_name):
        all_data = self.get_file_names_and_classes(data_dir_name)
        # print(all_data, data_dir_name)
        inter_count, preic_count = self.count_class_occurrences(all_data)
        print('Interictal/0 samples:', inter_count)
        print('Preictal/1 samples:', preic_count)

        # inter_count = inter_count - 8
        #preic_count = 140

        data_random_interictal = np.random.choice(
            all_data[all_data['class'] == self.INTERICTAL_CLASS],
            size=inter_count)

        # Take all preictal cases as they are scarce
        data_random_preictal = np.random.choice(
            all_data[all_data['class'] == self.PREICTAL_CLASS],
            size=preic_count)

        return data_random_interictal, data_random_preictal

    def merge_and_shuffle_selection(self, interictal_set, preictal_set):
        data_files = np.concatenate([interictal_set, preictal_set])
        data_files.dtype = interictal_set.dtype
        np.random.shuffle(data_files)
        return data_files

    def normalize(self, x):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        return x_scaled

    def get_X_from_files(self,
                         data_dir_base,
                         data_files,
                         train_data=True,
                         show_progress=True):

        eeg_data = []
        eeg_labels = []
        file_ids = []
        print(data_dir_base)

        total_files = len(data_files)

        for i, filename in enumerate(data_files):
            if show_progress and i % int(total_files / 1) == 0:
                print(u'%{}: Loading file {}'.format(
                    int(i * 100 / total_files), filename))

            try:
                mat_data = scipy.io.loadmat(
                    '/'.join([data_dir_base, filename.decode('UTF-8')]))
            except ValueError as ex:
                print(u'Error loading MAT file {}: {}'.format(filename,
                                                              str(ex)))
                continue

            eeg_image = mat_data['eeg_image']
            # pdb.set_trace()
            # eeg_image = eeg_image[:][200:800]

            # make sure we grab the right labels
            if train_data:
                label = self.get_class_from_name(filename)
                eeg_labels.append(label)
            eeg_data.append(eeg_image)
            file_ids.append(filename)

        return eeg_data, file_ids, eeg_labels

    '''
    def trans2image(self,channel_data):
        Z = np.zeros((self.samp_count,self.samp_count))
        fft_out = np.fft.fft(channel_data)
        for i in xrange(self.samp_count):
            for j in xrange(self.samp_count):
                Z[i,j] = np.exp(-np.abs(fft_out[i])/np.abs(fft_out[j]))
        Z = Z / np.max(Z)
        print 'done'
        return Z
    '''

    def subsample_data(self, channels_data, rate):
        channel_columns = channels_data.shape[0]
        timestep_rows = channels_data.shape[1]
        indices = np.arange(0, (timestep_rows - 1), rate)

        eeg_subsampl = np.zeros((channel_columns, len(indices)))
        for i in range(channel_columns):
            eeg_subsampl[i][:] = itemgetter(*indices)(channels_data[i])

        return eeg_subsampl

    def load_train_data(self, train_set_name):
        data_interictal, data_preictal = self.pick_random_observation(
                                                                      train_set_name)
        shuffled_dataset = self.merge_and_shuffle_selection(data_interictal,
                                                            data_preictal)

        print("Size of final training set", shuffled_dataset.shape)

        base_dir_train = self.path_to_all_datasets + '/' + train_set_name
        print("Data set directory==>", base_dir_train)
        X_train, labels, y_train = self.get_X_from_files(
            base_dir_train, shuffled_dataset['file'])
        y_train_or = shuffled_dataset['class']

        assert(np.array_equal(y_train, y_train_or))
        # print('filename', labels)
        # print('label', y_train)
        assert(len(X_train) == len(y_train))

        return X_train, y_train

    def load_test_data(self, test_set_name):
        base_dir_test = self.path_to_all_datasets + '/' + test_set_name
        print("Data set directory==>", base_dir_test)
        all_data = self.get_data_dir(test_set_name)
        #print("all data", all_data, len(all_data))

        X_test, file_ids, _ = self.get_X_from_files(
            base_dir_test, all_data,  False)
        return X_test, file_ids

    def next_training_batch(self,
                            X_train,
                            y_train):

        if(self.batch_index == len(X_train)):
            batch_xs = X_train[self.index_0: self.batch_index]
            batch_ys = y_train[self.index_0: self.batch_index]
            # Reset the data handler index
            self.index_0 = 0
            self.batch_index = self.batch_size
            return batch_xs, batch_ys

        batch_xs = X_train[self.index_0: self.batch_index]
        batch_ys = y_train[self.index_0: self.batch_index]
        self.index_0 += self.batch_size
        self.batch_index += self.batch_size
        if self.index_0 + self.batch_size > len(X_train):
            self.index_0 = len(X_train) - self.batch_size
        return batch_xs, batch_ys

    def get_evaluation_set(self,
                           X_train,
                           y_train,
                           eval_size):
        data_size = len(X_train) - 1

        rand_indices = random.sample(range(0, data_size), eval_size)
        X_train_sampl = itemgetter(*rand_indices)(X_train)
        y_train_sampl = itemgetter(*rand_indices)(y_train)
        print('Number of 1 cases', sum(y_train_sampl))
        return X_train_sampl, y_train_sampl
