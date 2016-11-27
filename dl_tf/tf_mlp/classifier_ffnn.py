#!/usr/bin/env python
"""Builds the booster network for MNIST weak classifiers.
The MNIST booster is a network which receives the output of two
weak classifiers and aims at performing a better classification
on the MNIST dataset by learning which weak network performs
better in what situation. The MNIST booster network input size
is therefore 20 (2 weak classifiers outputs, ecah has 10 output)
and its output size is 10 (number of classes). The number and sizes
of hidden layers (as well as their activation functions) are user-
defined.
"""
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import time
import pdb
from sys import stdout


class Classifier_Network:

    def __init__(self,
                 FLAGS,
                 input_size,
                 output_size,
                 hidden_layer_cnt = 1,
                 hidden_sizes = [100],
                 batch_size = 100,
                 hidden_act = tf.nn.relu,
                 output_act = tf.nn.softmax,
                 pos_weight = 1.):
        """Initializes a MNIST booster
        Args:
            hidden_layer_cnt: number of hidden layers,
            hidden_sizes: list of number of neurons in each hidden layer,
            batch_size: Size of the batch.
            hidden_acts: hidden layer activation function,
            output_act = output layer activation function
        """
        self._in_size    = input_size        # dimension of network input
        self._out_size   = output_size       # dimension of network output
        self._lcnt       = hidden_layer_cnt  # num of hidden layers
        self._hsize      = hidden_sizes # num of neurons in each hidden layer
        self._batch_size = batch_size   # num of samples in training batch
        self._hact       = hidden_act   # hidden layer activation function
        self._oact       = output_act   # output layer acitvation function
        self._pos_weight = pos_weight
        # define the network placeholders
        self._inputs_pl = tf.placeholder(tf.float32, shape=(None,
                                                            self._in_size))
        self._labels_pl = tf.placeholder(tf.int32, shape=(None))

        self._buildNet()
        self._sess = tf.Session()

        # private attributes
        self._FLAGS = FLAGS
        self._correct           = None
        self._cross_entropy     = None
        self._feed_dict         = None
        self._feed_dict_full    = None
        self._labels            = None
        self._loss              = None
        self._test_eval         = None
        self._train_op          = None
        self._saver             = tf.train.Saver()

    def _weightVariable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape,
                                      stddev=1.0 / math.sqrt(float(shape[0])),
                                      name='weights')
        return tf.Variable(initial)

    def _biasVariable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _nnLayer(self,
                  input_tensor,
                  input_dim,
                  output_dim,
                  layer_name,
                  act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the
        # graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self._weightVariable([input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = self._biasVariable([output_dim])
            with tf.name_scope('ridge_transform'):
                preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
            return activations

    def _buildNet(self):
        # Build the booster network.
        # input layer
        hidden = self._nnLayer(self._inputs_pl,
                                self._in_size,
                                self._hsize[0],
                                layer_name = 'input',
                                act = self._hact)
        # making hidden layers
        for i in range(1, self._lcnt):
            astr = 'hidden{0}'.format(i)
            hidden = self._nnLayer(hidden,
                                    self._hsize[i - 1],
                                    self._hsize[i],
                                    layer_name = astr,
                                    act = self._hact)
        # making output layer
        # NOTE: The output activation function is applied on the ridge
        # transform in the loss function
        with tf.name_scope('output_act'):
            weights = self._weightVariable([self._hsize[self._lcnt-1], self._out_size])
            biases  = self._biasVariable([self._out_size])
            self._logits  = tf.matmul(hidden, weights) + biases
            self._outputs = self._oact(self._logits)

    def setupLoss(self):
        """Calculates the loss from the logits and the labels.
        """
        self._labels = tf.to_float(self._labels_pl)
        self._cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                    self._logits,
                    self._labels,
                    self._pos_weight,
                    name = 'wentropy')
        # self._cross_entropy = \
        #         tf.nn.sparse_softmax_cross_entropy_with_logits(self._logits,
        #                                                        self._labels,
        #                                                        name='xentropy')
        self._loss = tf.reduce_mean(self._cross_entropy, name='xentropy_mean')

    def _setupTrainingOp(self, learning_rate):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            learning_rate: The learning rate to use for gradient descent.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(self._loss.op.name, self._loss)
        # Create the gradient descent optimizer with the given learning rate.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        self._train_op = optimizer.minimize(self._loss, global_step=global_step)

    def _setupEvaluation(self):
        """Evaluate the quality of the logits at predicting the label.
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        rounded_output = tf.round(self._outputs)
        self._correct = tf.equal(rounded_output, self._labels)
        # Return the number of true entries.
        self._test_eval = tf.reduce_sum(tf.cast(self._correct, tf.int32))

    def _fillFeedDict(self, dataset, full = False):
        """Fills the feed_dict for training the given step.
        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.(
        if full:
            inputs_feed = dataset.images
            labels_feed = dataset.labels
        else:
            inputs_feed, labels_feed = dataset.next_batch(self._batch_size)
        self._feed_dict = {
            self._inputs_pl: inputs_feed,
            self._labels_pl: labels_feed,
        }

    def _eval(self, dataset):
        """Runs one evaluation against the full epoch of data.
        Args:
            data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
        """
        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = dataset.num_examples // self._batch_size

        for step in xrange(steps_per_epoch):
            self._fillFeedDict(dataset)
            true_count += self._sess.run(self._test_eval,
                                        feed_dict=self._feed_dict)
        precision = 100 * true_count / dataset.num_examples
        # pdb.set_trace()
        print '\tNum examples: ', dataset.num_examples,\
              '\t Num correct: ', true_count,\
              '\t Precision  %', precision,'%'
        return precision

    def producePredictions(self, dataset):
        predictions = []
        for i in xrange(dataset.num_examples):
            predict = self._sess.run(self._outputs,
                                     feed_dict =
                                     {self._inputs_pl: np.reshape(
                                         dataset.images[i],(1,-1))})
            predictions.append(predict)
        return predictions



    def train(self, dataset, FLAGS):
        # Add the op to optimize
        self._setupTrainingOp(FLAGS.learning_rate)
        # Add the op to evaluate the classifier
        self._setupEvaluation()
        # Add the variable intializer Op.
        init = tf.initialize_all_variables()
        # Run the Op to initialize the variables
        self._sess.run(init)
        # calculate number of iteration per an epoche
        num_itr = int(round(dataset.train.num_examples / self._batch_size) + 1)
        # Start the training loop
        # epoche loops
        for epoch in xrange(FLAGS.epochs):
            start_time = time.time()
            # pdb.set_trace()
            for step in xrange(num_itr):
                self._fillFeedDict(dataset.train)
                _, cost = self._sess.run([self._train_op, self._loss],
                                         feed_dict=self._feed_dict)
            duration = time.time() - start_time
            print('Epoch %d took %.3f sec, cost: %.8f' % (epoch,duration,cost))
            # Evaluate against the training set.
            # print('\tTraining Data Eval:'),
            # stdout.flush()
            # self._eval(dataset.train)
            # Evaluate against the validation set.
            # print '\tValidation Data Eval:',
            val_accuracy = self._eval(dataset.validation)
            self._fillFeedDict(dataset.validation)
            val_cost = self._sess.run(self._loss,self._feed_dict)
            if epoch == 0:
                self._saver.save(self._sess,
                                 self._FLAGS.model_dir)
                best_accuracy = val_accuracy
                best_cost = val_cost
            if val_cost < best_cost:
                best_cost = val_cost
                best_accuracy = val_accuracy
                self._saver.save(self._sess,
                                 self._FLAGS.model_dir)
            print '\tBest cost so far: ', best_cost, ', best accuracy so far:', best_accuracy
            # Evaluate against the test set.
            # print('\tTest Data Eval:'),
            # self._eval(dataset.test)

    def fullEval(self, dataset):
        """Runs the full evaluation against the entire dataset.
        Args:
            data_set: The set of images and labels to evaluate.
        """
        # Fill the placeholders with the given dataset
        self._fillFeedDict(dataset, full = True)
        # Get the loss for all samples
        loss_values = self._sess.run(self._cross_entropy,
                                    feed_dict=self._feed_dict)
        # True counts
        predictions = self._sess.run(self._correct,
                                    feed_dict=self._feed_dict)
        correct_class_idx = np.where(predictions)
        misclassified_idx = np.where(predictions == False)
        return correct_class_idx[0], misclassified_idx[0], loss_values

    def calcOutputs(self, dataset):
        """ Calculate the softmax output of the logits over the
        entire dataset.
        Args:
            dataset: The set of images and labels.
        """
        # print('Output calculation:')
        self._fillFeedDict(dataset, full = True)
        output_values = self._sess.run(self._outputs,
                                      feed_dict = self._feed_dict)
        return output_values

    def extractIndices(self, ds):
        # extract the indices and costs
        cid, wid, cst = self.fullEval(ds)
        adict = dict(correct_ind=cid.copy(),
                     wrong_ind=wid.copy(),
                     cost=cst.copy())
        return adict

    def load(self):
        self._saver.restore(self._sess,self._FLAGS.model_dir )

if __name__ == '__main__':

        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string('data_dir', 'data', 'Directory to put the training data.')
        flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
        flags.DEFINE_integer('num_steps', 2000, 'Number of steps to run trainer.')
        ds = XDataset('data')
        with tf.Graph().as_default():
            # create and train a weack classifier
            print(' Test a feedforward neural network on clasifying MNIST:')
            wnet = Classifier_Network(input_size = 28 * 28,
                                      output_size = 10,
                                      hidden_layer_cnt = 1,
                                      hidden_sizes = [100],
                                      batch_size = 100,
                                      hidden_act = tf.nn.relu,
                                      output_act = tf.nn.softmax)
            wnet.setupLoss()
            wnet.train(ds, FLAGS)
