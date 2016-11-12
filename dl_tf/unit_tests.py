#!/usr/bin/env python2
import unittest
import numpy as np
import matplotlib.pyplot as plt
from data_loader import SeizureDataset
import pdb

class DataLoaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = SeizureDataset(1000,
                                'train_1_dummy',
                                'test_1_dummy')

    def test_label(self):

        all_data = ['1_843_0.mat', '1_58_1.mat', '1_919_0.mat']

        self.assertEqual(self.ds.get_class_from_name(all_data[0]), 0)
        self.assertEqual(self.ds.get_class_from_name(all_data[1]), 1)
        self.assertEqual(self.ds.get_class_from_name(all_data[2]), 0)

    def test_class_occurrence_counter(self):
        print('Test occurrence counter')
        all_data = ['1_843_0.mat', '1_58_1.mat', '1_919_0.mat']

        file_with_class = np.array(
            [(mat_file, self.ds.get_class_from_name(mat_file))
             for mat_file in all_data],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])

        inter, preic = self.ds.count_class_occurrences(file_with_class)
        self.assertEqual(inter, 2, 'Wrong number of interictal case')
        self.assertEqual(preic, 1, 'Wrong number of preictal case')

    def test_rand_observation_loader(self):
        all_data = [
            '1_843_0.mat',
            '1_58_1.mat',
            '1_919_0.mat',
            '1_580_0.mat',
            '1_631_0.mat',
            '1_93_1.mat',
            '1_967_0.mat',
            '1_788_0.mat',
            '1_140_0.mat',
            '1_619_0.mat',
            ]

        file_with_class = np.array(
            [(mat_file, self.ds.get_class_from_name(mat_file))
             for mat_file in all_data],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])

        inter, preic = self.ds.count_class_occurrences(file_with_class)
        self.assertEqual(inter, 8, 'Wrong number of interictal case')
        self.assertEqual(preic, 2, 'Wrong number of preictal case')
        rand_interictal = np.random.choice(
            file_with_class
            [file_with_class['class'] == self.ds.INTERICTAL_CLASS],
            size=inter)

        rand_preictal = np.random.choice(
            file_with_class[file_with_class['class'] == self.ds.PREICTAL_CLASS],
            size=preic)
        self.assertEquals(
            len(rand_interictal),
            inter,
            'Size of loaded interictal does not match')
        self.assertEquals(
            len(rand_preictal),
            preic,
            'Size of loaded preictal does not match')

    def test_check_ids_match(self):

        all_data = [
            '1_843_0.mat',
            '1_58_1.mat',
            ]

        file_with_class = np.array(
            [(mat_file, self.ds.get_class_from_name(mat_file))
             for mat_file in all_data],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])

        ids = file_with_class['file']
        self.assertEqual(ids[0], '1_843_0.mat')
        self.assertEqual(ids[1], '1_58_1.mat')

    def test_data_loader(self):
        train_set_name = 'train_1_dummy'
        data_interictal, data_preictal = self.ds.pick_random_observation(
                                        train_set_name)
        shuffled_dataset = self.ds.merge_and_shuffle_selection(data_interictal,
                                                               data_preictal)

        print("test data loader")
        # print(shuffled_dataset)
        #eeg_label = shuffled_dataset['class']
        #print("Labels", eeg_label)

        base_dir_train = self.ds.path_to_all_datasets + '/' + train_set_name
        #print("Data set directory==>", base_dir_train)
        eeg_data, _ = self.ds.get_X_from_files(
            base_dir_train, shuffled_dataset['file'],
            self.ds.input_subsample_rate)

        data_indx = 0
        chan = 1
        plt.plot(eeg_data[data_indx][chan, :])
        # plt.show()

    def test_next_batch(self):
        all_data = [
            '1_843_0.mat',
            '1_58_1.mat',
            '1_919_0.mat',
            '1_580_0.mat',
            '1_631_0.mat',
            '1_93_1.mat',
            '1_967_0.mat',
            '1_788_0.mat',
            '1_140_0.mat',
            '1_619_0.mat',
            ]

        file_with_class = np.array(
            [(mat_file, self.ds.get_class_from_name(mat_file))
             for mat_file in all_data],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])

        X_train = file_with_class['file']
        y_labels = file_with_class['class']

        batch_size = 2
        self.ds.set_batch_size(batch_size)

        total_batch = int(len(X_train)/batch_size)

        self.assertEqual(total_batch, 5, 'Wrong total batch size')
        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)

        self.assertEqual(len(batch_xs), batch_size, 'Wrong x_train batch size')

        self.assertEqual(batch_xs[0], '1_843_0.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_xs[1], '1_58_1.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_ys[0], 0, 'Label did not match the batch file')
        self.assertEqual(batch_ys[1], 1, 'Label did not match the batch file')

        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)

        self.assertEqual(len(batch_xs), batch_size, 'Wrong x_train batch size')

        self.assertEqual(batch_xs[0], '1_919_0.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_xs[1], '1_580_0.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_ys[0], 0, 'Label did not match the batch file')
        self.assertEqual(batch_ys[1], 0, 'Label did not match the batch file')

        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)

        self.assertEqual(len(batch_xs), batch_size, 'Wrong x_train batch size')

        self.assertEqual(batch_xs[0], '1_631_0.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_xs[1], '1_93_1.mat', 'Grabbed the wrong file')
        self.assertEqual(batch_ys[0], 0, 'Label did not match the batch file')
        self.assertEqual(batch_ys[1], 1, 'Label did not match the batch file')

    def test_next_batch_completion(self):
        all_data = [
            '1_843_0.mat',
            '1_58_1.mat',
            '1_919_0.mat',
            '1_580_0.mat'
            ]

        file_with_class = np.array(
            [(mat_file, self.ds.get_class_from_name(mat_file))
             for mat_file in all_data],
            dtype=[('file', '|S16'),
                   ('class', 'float32')])

        X_train = file_with_class['file']
        y_labels = file_with_class['class']

        batch_size = 2
        self.ds.set_batch_size(batch_size)

        total_batch = int(len(X_train)/batch_size)

        self.assertEqual(total_batch, 2, 'Wrong total batch size')
        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)
        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)

        self.assertEqual(self.ds.index_0, batch_size, 'Wrong index update')
        # In this case batch*2 == len(X_train) since its the final batch
        self.assertEqual(
            self.ds.batch_index,
            len(X_train),
            'Wrong batch index update')
        batch_xs, batch_ys = self.ds.next_training_batch(X_train,
                                                         y_labels)

        print('Size of index 0', self.ds.index_0)
        self.assertEqual(
            self.ds.index_0,
            batch_size,
            'Next batch generation failed')
        self.assertEqual(
            self.ds.batch_index,
            batch_size*2,
            'Next batch generation failed')
        self.assertIsNotNone(batch_xs)
        self.assertIsNotNone(batch_ys)

    def test_evaluation_loader(self):
        self.ds.input_subsample_rate = 240000/160
        eval_size = 2
        X_train, y_train = self.ds.load_train_data('train_1_dummy')
        X_train_eval_sampl, y_train_eval_sampl = self.ds.get_evaluation_set(
                                                X_train,
                                                y_train,
                                                eval_size)

        self.assertEqual(len(X_train_eval_sampl), eval_size, 'Wrong eval size')
        self.assertEqual(len(y_train_eval_sampl), eval_size, 'Wrong eval size')
        self.assertIsInstance(y_train_eval_sampl[0], np.float32)
        # If this fails, double check the subsample rate
        self.assertEqual(
            X_train_eval_sampl[0].shape,
            (16,
             self.ds.input_subsample_rate),
            'Size of the subsampled data does not match the original data')


    def test_data_sampler(self):

        dummy_matrix = np.array(
            [[-0.13130315,  -3.13130307,  -9.13130283, 24.86869621,
                23.86869621,  28.86869621],
             [2.33613181,  5.33613157,  18.33613205, 30.33613205,
                36.33613205,  28.33613205],
             [-26.15572548, -35.15572357, -39.15572357, 3.84427452,
                4.84427452,   4.84427452]])
        dummy_matrix_downsampl = np.array(
                [[-0.13130315, -9.13130283,  23.86869621],
                 [2.33613181,  18.33613205,  36.33613205],
                 [-26.15572548, -39.15572357,   4.84427452]])

        downsample_rate = 2
        subsampled_mat = self.ds.subsample_data(dummy_matrix, downsample_rate)
        self.assertEqual(
            subsampled_mat.shape, (3, 3),
            'Mismatch in downsampled size')
        self.assertTrue(np.array_equal(subsampled_mat, dummy_matrix_downsampl),
                        'Downsampled matrices do not match')

if __name__ == '__main__':
    unittest.main()
