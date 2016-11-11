import pandas as pd
import os

path_to_all_datasets = os.path.abspath(os.path.join(
                        '..', 'data_dir/Kaggle_data/data'))
submission_file_name = path_to_all_datasets + "/sample_submission.csv"
submission_file = pd.read_csv(submission_file_name)
pred_dir = path_to_all_datasets + "/Pred_1/"
test_files = os.listdir(pred_dir)

for file_name in test_files:
        df = pd.read_csv(pred_dir + file_name)
        df.columns = ['File', 'Class']
        submission_file = pd.merge(submission_file, df, how='left', on = ['File'], suffixes=['_1','_2'])
        submission_file['Class_2'] = submission_file['Class_2'].fillna(0)
        submission_file['Class'] = submission_file['Class_1'] + submission_file['Class_2']
        submission_file.drop(['Class_1', 'Class_2'], axis=1, inplace=True)
submission_file['Class'] = submission_file['Class']
submission_file.to_csv(path_to_all_datasets + "/Submission_final.csv", index=False)

