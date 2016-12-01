#!/usr/bin/env python
"""Builds a simple neural network experimenting with the following techniques :
    - Regularization
    - weighted cross entropy cost function to make up for imbalanced data
    - Cost pertubation
"""
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import time
import pdb

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
                 keep_prob = 1.,
                 pos_weight = 1.):
        """Initializes the MLP classifier
        Args:
            FLAGS: input arguments
            input_size: inputs size of the network
            output_size: number of classes
            hidden_layer_cnt: number of hidden layers (excluding the output)
            hidden_sizes: size of hidden layers (excluding the output)
            batch_size: number of samples in the training batch
            hidden_act: activation function for the hidden neurons
            output_act: activation function for the network output
            keep_prob: 1 - dropout probability
            pos_weight: positive coefficient in the loss function
        """
        self._in_size    = input_size
        self._out_size   = output_size
        self._lcnt       = hidden_layer_cnt
        self._hsize      = hidden_sizes
        self._batch_size = batch_size
        self._hact       = hidden_act
        self._oact       = output_act
        self._pos_weight = pos_weight
        self._l2_coeff = 0.0
        self._dropout = False;
        if keep_prob < 1.:
            self._dropout = True
            self._keep_prob = keep_prob
        # define the network placeholders
        self._inputs_pl = tf.placeholder(tf.float32,
                                         shape=(None, self._in_size))
        self._labels_pl = tf.placeholder(tf.int32,
                                         shape=(None))
        if self._dropout:
            self._dropout_pl = tf.placeholder(tf.float32)

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
                                      stddev=1. / math.sqrt(float(shape[0])),
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
                 set_dropout = False,
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
            if (self._dropout) & (set_dropout):
                return tf.nn.dropout(activations, self._dropout_pl)
            else:
                return activations

    def _buildNet(self):
        # input layer
        hidden = self._nnLayer(self._inputs_pl,
                               self._in_size,
                               self._hsize[0],
                               layer_name = 'input',
                               set_dropout = True,
                               act = self._hact)
        # making hidden layers
        for i in range(1, self._lcnt):
            astr = 'hidden{0}'.format(i)
            hidden = self._nnLayer(hidden,
                                   self._hsize[i - 1],
                                   self._hsize[i],
                                   layer_name = astr,
                                   set_dropout = True,
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

        # Adding l2 regularization
        all_weights = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in all_weights]) * self._l2_coeff
        self._loss_raw = tf.reduce_mean(self._cross_entropy, name='xentropy_mean')
        self._loss = self._loss_raw + lossL2

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
            self._labels_pl: labels_feed
        }
        if self._dropout:
            self._feed_dict.update({self._dropout_pl: self._keep_prob})

    def _eval(self, dataset):
        """Runs one evaluation against the full epoch of data.
        Args:
            data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
        """
        true_count = 0  # Counts the number of correct predictions.

        for i in xrange(dataset.num_examples):
            afeed_dict = {
                self._inputs_pl: np.reshape(dataset.images[i],(1,-1)),
                self._labels_pl: dataset.labels[i]
            }
            if self._dropout:
                afeed_dict.update({self._dropout_pl: self._keep_prob})
            true_count += self._sess.run(self._test_eval,
                                        feed_dict=afeed_dict)
        precision = 100 * true_count / dataset.num_examples
        # pdb.set_trace()
        print '\tNum examples: ', dataset.num_examples,\
              '\t Num correct: ', true_count,\
              '\t Precision  %', precision,'%'
        return precision

    def producePredictions(self, dataset):
        predictions = []
        for i in xrange(dataset.num_examples):
            afeed_dict ={self._inputs_pl: np.reshape(
                dataset.images[i],(1,-1))}
            if self._dropout:
                afeed_dict.update({self._dropout_pl: self._keep_prob})
            predict = self._sess.run(self._outputs,
                                     feed_dict = afeed_dict)
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

        # Start the training loop
        # epoche loops
        local_min_cnt = 0
        perturb = 0
        costs = np.zeros(FLAGS.epochs)
        max_local_mins = FLAGS.max_localmin_iters
        for epoch in xrange(FLAGS.epochs):
            # calculate number of iteration per an epoche
            num_itr = int(round(dataset.train.num_examples / self._batch_size) + 1)
            start_time = time.time()
            for step in xrange(num_itr):
                self._fillFeedDict(dataset.train)
                _, cost = self._sess.run([self._train_op, self._loss],
                                         feed_dict=self._feed_dict)

            val_accuracy = self._eval(dataset.validation)
            self._fillFeedDict(dataset.validation,True)
            val_cost = self._sess.run(self._loss_raw,self._feed_dict)
            costs[epoch] = val_cost

            duration = time.time() - start_time
            print('Epoch %d took %.3f sec, cost: %f' % (epoch,duration,val_cost))
            if epoch == 0:
                self._saver.save(self._sess,
                                 self._FLAGS.model_dir)
                best_accuracy = val_accuracy
                best_cost = val_cost

            if val_cost < best_cost:
                local_min_cnt = 0
                perturb = 0
                best_cost = val_cost
                best_accuracy = val_accuracy
                self._saver.save(self._sess,
                                 self._FLAGS.model_dir)
                print('#############################\n')
                print('# Best validation detected! #\n')
                print('#############################\n')
            else:
                local_min_cnt += 1

            print '\tBest cost so far: ', best_cost,
            print ', corresponding accuracy: ', best_accuracy,
            print ', local min iterations: ', local_min_cnt

            if local_min_cnt == max_local_mins:
                local_min_cnt = 0
                perturb += 1
                self._l2_coeff = 1 * perturb
                self.setupLoss()
            if perturb == FLAGS.max_perturbs:
                return costs
        return costs


    def _perturbWeights(self,
                        perturb_mech = 'weight_wise',
                        random_step = 'uniform',
                        magnitude = 0.1):
        #get list of all trainable weights (and biases)
        all_weights = tf.trainable_variables()
        for i in xrange(len(all_weights)):
            # get the weight values
            weight = self._sess.run(all_weights[i])
            # random step
            if random_step == 'uniform':
                perturbation = np.random.rand(*weight.shape)
            # check perturbnation mechanism to apply perturbation
            if perturb_mech == 'weight_wise':
                self._sess.run(all_weights[i].assign_add(magnitude * np.multiply(perturbation,
                                                                                 weight)))

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
        misclassified_idx = np.where(predictions is False)
        return correct_class_idx[0], misclassified_idx[0], loss_values

    def calcOutputs(self, dataset):
        """ Calculate the softmax output of the logits over the
        entire dataset.
        Args:
            dataset: The set of images and labels.
        """
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

