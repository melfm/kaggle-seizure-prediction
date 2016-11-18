import scipy.io
import numpy as np
import os

def get_data_dir(train_x):
    path_to_all_datasets = os.path.abspath(os.path.join(
        '..', 'data_dir/Kaggle_data/data/'))
    path_to_dataset = os.path.join(path_to_all_datasets, train_x)

    print 'Loading data set:\n', path_to_dataset, '\n'

    data_files = os.listdir(path_to_dataset)
    for mat_file in data_files:
        assert(mat_file.endswith('.mat')) == True
    return data_files


def get_class_from_name(name):
    try:
        return float(name[-5])
    except:
        return 0.0


def get_file_names_and_classes(data_dir_name):
    all_data = get_data_dir(data_dir_name)

    file_with_class = np.array([(mat_file, get_class_from_name(mat_file))
                                for mat_file in all_data],
                               dtype=[('file', '|S16'), ('class', 'float32')])
    return file_with_class


def remove_zero_matrices(data_dir_base,
                         data_files,
                         show_progress=True):

    print(data_dir_base)

    total_files = len(data_files)
    removed_files = 0

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

        # Gets a 16x240000 matrix => 16 channels reading data for 10 minutes at
        # 400Hz
        channels_data_nn = mat_data['dataStruct'][0][0][0].transpose()

        if (np.all(channels_data_nn == 0)):
            # Remove the file
            file_to_remove = data_dir_base + '/' + filename
            assert(os.path.isfile(file_to_remove))
            os.remove(file_to_remove)
            assert(os.path.isfile(file_to_remove), False)
            print('Removed ', filename)
            removed_files += 1
    print('Removed ', removed_files, 'files')


training_set = ['train_1', 'train_2', 'train_3']

for i in range(len(training_set)):
    base_dir_train = os.path.abspath(os.path.join(
            '..', 'data_dir/Kaggle_data/data/'))
    base_dir_train = base_dir_train + '/' + training_set[i]
    all_data = get_file_names_and_classes(training_set[i])
    remove_zero_matrices(base_dir_train, all_data['file'])
