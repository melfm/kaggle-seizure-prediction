# Melbourne University AES/MathWorks/NIH Seizure Prediction

Neural network solution using tensorflow. Three architectures explored, MLP, CNN and RNN.

## Dependencies:
- Tensorflow
- numpy
- Scikit-learn (optional)
- Matlab (to run scripts to pre-process the data)

## Usage:
- Expects the data under the dir: data_dir/Kaggle_data. This includes sample_submission and train_and_test_data_labels_safe (which was used to throw away *bad* data).
- Before pre-processing the data, it is a good idea to run the data_cleanser script to remove the all-zero files, although during the pre-processing step the matlab script should also handle this.


Tested on Ubuntu 14.04, tensorflow 0.11.0rc1, GPU:TitanX.

## Performance and accuracy:
- Training speed differs for each architecture, a few hours max.
- The best AUC achieved with MLP ~0.68-0.69

## TODO:
- Testing (operations such as weight perturbation is one example)
- CNN architecture contains a bug in the cost (this was fixed in MLP), which causes the cost to jump.

