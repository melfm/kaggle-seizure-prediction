#! /usr/bin/env/ python2
from scipy.signal import resample
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import pdb
import os
from sklearn import preprocessing


class SeizureDataset:

    #####################
    INTERICTAL_CLASS = 0
    PREICTAL_CLASS = 1
    #####################
    train_class_ratio = 0.24

    path_to_all_datasets = os.path.abspath(os.path.join(
                        '../..', 'data_dir/Kaggle_data/data'))

    safe_label_files = 'train_and_test_data_labels_safe.csv'

    def __init__(self,
                 input_subsample_rate=1000,
                 train_set='train_1',
                 test_set='test_1_new'):
        self.input_subsample_rate = input_subsample_rate
        self.train_set = train_set
        self.test_set = test_set

    def get_data_dir(self):

        path_to_dataset = os.path.join(
    self.path_to_all_datasets, self.train_set)
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
        safe_labels = os.path.join(self.path_to_all_datasets,self.safe_label_files)
        df_safe=pd.read_csv(safe_labels)

        ignored_files=df_safe.loc[df_safe['safe'] == 0]
        ignored_files=ignored_files['image'].tolist()
        ignored_files.append('1_45_1.mat')
        print('Ignoring these files:', ignored_files)
        all_data=self.get_data_dir(data_dir_name)

        file_with_class=np.array([(mat_file, self.get_class_from_name(mat_file))
                                    for mat_file in all_data if mat_file not in ignored_files],
                                   dtype=[('file', '|S16'), ('class', 'float32')])
        return file_with_class


    def pick_random_observation(self, data_dir_name):
        all_data = self.get_file_names_and_classes(data_dir_name)
        print(all_data, data_dir_name)
        inter_count, preic_count = self.count_class_occurrences(all_data)
        print('Interictal/0 samples:', inter_count)
        print('Preictal/1 samples:', preic_count)

        # data_random_interictal = np.random.choice(
        #    all_data[all_data['class'] == INTERICTAL_CLASS],
        #    size=round(preic_count + (train_class_ratio * inter_count)))
        # Just load everything since we have a more balanced
        # dataset and penalizing non-positives more
        data_random_interictal = np.random.choice(
            all_data[all_data['class'] == self.INTERICTAL_CLASS],
            size=inter_count)

        # Take all preictal cases as they are scarce
        data_random_preictal=np.random.choice(
            all_data[all_data['class'] == self.PREICTAL_CLASS],
            size=preic_count)

        return data_random_interictal, data_random_preictal


    def merge_and_shuffle_selection(self, interictal_set, preictal_set):
        data_files=np.concatenate([interictal_set, preictal_set])
        data_files.dtype=interictal_set.dtype
        np.random.shuffle(data_files)
        return data_files

    def normalize(df):
        x=df.values
        min_max_scaler=preprocessing.MinMaxScaler()
        x_scaled=min_max_scaler.fit_transform(x)
        return x_scaled

    def get_X_from_files(
    self,
    data_dir_base,
    data_files,
    sub_sample_rate,
     show_progress=True):

        eeg_data=[]
        file_ids=[]
        print(data_dir_base)

        total_files=len(data_files)

        for i, filename in enumerate(data_files):
            if show_progress and i % int(total_files / 1) == 0:
                print(u'%{}: Loading file {}'.format(
                    int(i * 100 / total_files), filename))

            try:
                mat_data=scipy.io.loadmat(
                    '/'.join([data_dir_base, filename.decode('UTF-8')]))
            except ValueError as ex:
                print(
    u'Error loading MAT file {}: {}'.format(
        filename, str(ex)))
                continue

            # Gets a 16x240000 matrix => 16 channels reading data for 10 minutes at
            # 400Hz
            channels_data_nn=mat_data['dataStruct'][0][0][0].transpose()
            # Resamble each channel to get data at 100Hz
            channels_data_nn=resample(
    channels_data_nn, sub_sample_rate, axis=1, window=400)

            # drop if nan
            df=pd.DataFrame(channels_data_nn, index=None)
            df=df.dropna()
            # Normalize
            channels_data = self.normalize(df)
            eeg_data.append(channels_data)
            file_ids.append(filename)

        return eeg_data, file_ids


    def test_data_loader(self, train_set_name):
        data_interictal, data_preictal = self.pick_random_observation(train_set_name)
        shuffled_dataset = self.merge_and_shuffle_selection(data_interictal,
                                                       data_preictal)

        print("test data loader")
        print(shuffled_dataset)
        eeg_label=shuffled_dataset['class']
        print("Labels", eeg_label)
        base_dir_train=os.path.abspath(os.path.join(
                        '../..', 'data_dir/Kaggle_data/data/'))
        base_dir_train=base_dir_train + '/' + train_set_name
        print("Data set directory==>", base_dir_train)
        eeg_data, _= self.get_X_from_files(
                            base_dir_train, shuffled_dataset['file'], self.input_subsample_rate)

        data_indx=3
        chan=1
        plt.plot(eeg_data[data_indx][chan, :])
        plt.show()
