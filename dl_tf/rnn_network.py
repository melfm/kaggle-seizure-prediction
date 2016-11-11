#!/usr/bin/env python
"""Tensorflow Recurrent network.
Implements the inference/loss/training pattern for model building.
1. Builds the model as far as is required for running the network
forward to make predictions.
2. Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
"""
from __future__ import division

from tensorflow.python.ops import rnn_cell
import tensorflow as tf
import numpy as np
import time
import math
import os
#import pdb


class SeizureClassifier:

    def __init__(self,
                 input_dim=16,
                 input_timestep=1000,
                 output_classes=1,
                 batch_size=20,
                 hidden1_units=100,
                 pos_weight=2
                 ):
        """ Initialize the network variables
        Args:
            input_dim: Number of channels in the data i.e. column data
            input_timestep: Number of signal time steps after subsampling
            output_classes: Number of outcomes, 0 or 1
            hidden1_units: Size of the first hidden layer
        """
        self.input_dim = input_dim
        self.input_timestep = input_timestep
        self.output_classes = output_classes
        self.batch_size = batch_size
        self.hidden1_units = hidden1_units
        self.pos_weight = pos_weight

        self.num_threads = 5

        self.x_pl = tf.placeholder(
            tf.float32, [None, self.input_timestep, self.input_dim],
            name='x-input')
        self.y_pl = tf.placeholder(tf.float32,
                                   [None, self.output_classes],
                                   name='y-input')

        self._build_net()
        # self.sess = tf.Session(config=tf.ConfigProto(
        #    intra_op_parallelism_threads=self.num_threads))
        self.sess = tf.Session()
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        self.loss = None
        self.cross_entropy = None
        self.train_op = None
        self.test_eval = None
        self.feed_dict = None

    def _weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape,
                                      stddev=1.0 / math.sqrt(float(shape[0])),
                                      name='weights')
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.random_normal(shape=shape)
        return tf.Variable(initial)

    def _variable_summaries(self, var, name):
        """Attach summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def _rnn_layer(self,
                   input_tensor,
                   input_dim,
                   output_dim,
                   layer_name):

        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self._weight_variable([input_dim, output_dim])
                self._variable_summaries(weights, 'h1/weights')

            with tf.name_scope('biases'):
                biases = self._bias_variable([output_dim])
                self._variable_summaries(biases, 'b1/biases')

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps'=>
            # Tensors list of shape (batch_size, n_input)

            # Permuting batch_size and n_steps:
            input_tensor = tf.transpose(input_tensor, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            input_tensor = tf.reshape(input_tensor, [-1, self.input_dim])
            # Split to get a list of 'n_steps' tensors of shape (batch_size,
            # n_input)
            input_tensor = tf.split(0, self.input_timestep, input_tensor)
            # Define a lstm cell with tensorflow
            rnncell = rnn_cell.BasicLSTMCell(
                self.hidden1_units, forget_bias=1.0, state_is_tuple=True)
            # rnncell = rnn_cell.BasicRNNCell(
            #          hidden1_neurons)
            # Get lstm cell output
            outputs, states = tf.nn.rnn(rnncell, input_tensor, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights) + biases

    def _build_net(self):
        # Build the classifier network
        self.logits = self._rnn_layer(self.x_pl,
                                      self.hidden1_units,
                                      self.output_classes,
                                      'hidden1')

    def setup_loss(self):
        """Calculates the loss from the logits and the labels."""
        self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                    self.logits, self.y_pl, self.pos_weight)
        self.loss = tf.reduce_mean(
            self.cross_entropy,
            name='weighted_entropy_mean')

    def setup_training_op(self, learning_rate):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            learning_rate: The learning rate to use for gradient descent.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary('cross entropy', self.loss)
        # Create the optimizer
        optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def do_eval(self, X_test, y_train):
        true_count = 0  # Counts the number of correct predictions
        num_examples = len(X_test)
        for i in range(num_examples):
            batch_xs_test = X_test[i]
            batch_xs = np.reshape(
                batch_xs_test, (1, -1, batch_xs_test.shape[0]))
            batch_y_test = int(y_train[i])
            batch_y_aslist = [batch_y_test]
            batch_y = np.reshape(batch_y_aslist, (1, len(batch_y_aslist)))
            dic = {self.x_pl: batch_xs, self.y_pl: batch_y}
            true_count += round(self.sess.run(tf.sigmoid(self.logits),
                                              feed_dict=dic)) == batch_y_test
            # pdb.set_trace()
        precision = true_count / num_examples
        print('Num examples: ', num_examples,
              'Num correct: ', true_count,
              'Precision  %', precision, '%')

    def do_train(self, data_handler, X_train, y_labels, FLAGS):
        # add the op to optimize
        self.setup_training_op(FLAGS.learning_rate)
        # Add the op tp evaluate the classifier

        # Add the variable initializer op.
        init = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                              self.sess.graph)
        # Run the op to initialize the variables
        self.sess.run(init)
        # Start the training iterations
        for epoch in xrange(FLAGS.epochs):
            start_time = time.time()
            total_batch = len(X_train)/self.batch_size
            for i in range(int(total_batch)):
                batch_xs = None
                batch_ys = None
                batch_xs, batch_ys = data_handler.next_training_batch(
                                    X_train,
                                    y_labels)
                batch_xs_tensor = np.reshape(batch_xs,
                                             (self.batch_size,
                                                 self.input_timestep,
                                                 self.input_dim))
                batch_ys_tensor = np.reshape(batch_ys,
                                             (self.batch_size,
                                                 self.output_classes))

                summary, _, c = self.sess.run([merged, self.train_op, self.loss],
                                              feed_dict={self.x_pl: batch_xs_tensor,
                                                         self.y_pl: batch_ys_tensor})
                train_writer.add_summary(summary, epoch)
            print("Epoch:", '%04d' %
                  (epoch + 1), "cost=", "{:.9f}".format(c))

            if ((epoch+1) % 5 == 0) or (epoch+1) == FLAGS.epochs \
                    or FLAGS.eval_net:
                print("Evaluation after %d epochs" % epoch)
                eval_size = 2
                if(FLAGS.eval_rand):
                    X_train_eval_sampl, y_train_eval_sampl = data_handler.get_evaluation_set(
                                                        X_train, y_labels, eval_size)

                    self.do_eval(X_train_eval_sampl, y_train_eval_sampl)
                else:
                    self.do_eval(X_train[0:eval_size], y_labels[0:eval_size])
            # Reset the data handler index
            #data_handler.index_0 = 0
            #data_handler.batch_index = self.batch_size

        print("Optimization Finishes!")
        # Do training here
        duration = time.time() - start_time
        print('Training duration: %.3f sec', duration)
        train_writer.close()
        # Save the final model to disk
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Saving the trained model')
        print('-------------------------')
        save_path = self.saver.save(
            self.sess, (FLAGS.model_dir + FLAGS.train_set + ".ckpt"))
        print('Model saved ->', save_path)

    def predict(self, X_test, FLAGS):
        # Restore model weights from previously saved model
        self.saver.restore(
            self.sess,
            (FLAGS.model_dir + FLAGS.train_set + ".ckpt"))
        predictions = []
        for i in range(len(X_test)):
            test_x = X_test[i]
            test_x_tensor = np.reshape(test_x,
                                       (1, -1, test_x.shape[0]))
            dic = {self.x_pl: test_x_tensor}
            pred = self.sess.run(tf.sigmoid(self.logits), feed_dict=dic)
            predictions.append(pred)
        return predictions
