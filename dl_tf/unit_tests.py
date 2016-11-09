#!/usr/bin/env python2
import unittest
import numpy as np
import matplotlib.pyplot as plt
from data_loader import SeizureDataset


class DataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = SeizureDataset(1000,
                                'train_1_dummy',
                                'test_1_dummy')

    def test_label(self):
        print('Test observation loader')

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
        print('Test random observation loader')
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
            file_with_class[file_with_class['class'] == self.ds.INTERICTAL_CLASS],
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
        self.assertEqual(ids[0],'1_843_0.mat')
        self.assertEqual(ids[1],'1_58_1.mat')



    def test_data_loader(self):
        train_set_name = 'train_1_dummy'
        data_interictal, data_preictal = self.ds.pick_random_observation(
                                        train_set_name)
        shuffled_dataset = self.ds.merge_and_shuffle_selection(data_interictal,
                                                            data_preictal)

        print("test data loader")
        print(shuffled_dataset)
        eeg_label = shuffled_dataset['class']
        print("Labels", eeg_label)

        base_dir_train = self.ds.path_to_all_datasets + '/' + train_set_name
        print("Data set directory==>", base_dir_train)
        eeg_data, _ = self.ds.get_X_from_files(
            base_dir_train, shuffled_dataset['file'],
            self.ds.input_subsample_rate)

        data_indx = 3
        chan = 1
        plt.plot(eeg_data[data_indx][chan, :])
        plt.show()


if __name__ == '__main__':
    unittest.main()
