#!/usr/bin/env/ python2
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
import pandas as pd
import pdb


class SeizureClassifier:

    def __init__(self,
                 FLAGS,
                 input_dim=300,
                 output_dim=1
                 ):
        """ Initialize the network variables
        Args:
            input_dim: Number of channels in the data i.e. column data
            input_timestep: Number of signal time steps after subsampling
            output_classes: Number of outcomes, 0 or 1
            hidden1_units: Size of the first hidden layer
        """
        self.input_dim = input_dim
        self.input_timestep = FLAGS.input_timestep
        self.output_dim = output_dim
        self.batch_size = FLAGS.batch_size
        self.pos_weight = FLAGS.pos_weight

        self.num_threads = 5

        self.x_pl = tf.placeholder(
            tf.float32, [None, self.input_timestep, self.input_dim],
            name='x-input')
        self.y_pl = tf.placeholder(tf.float32,
                                   [None, self.output_dim],
                                   name='y-input')

        self.x_pl = tf.placeholder(
            tf.float32,
            shape=[
                None,
                self.input_dim , self.input_timestep, 16])
        self.y_pl = tf.placeholder(tf.float32, shape=[None, self.output_dim])
        self.keep_prob = tf.placeholder(tf.float32)

        self._cnn_network()
        self.sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=self.num_threads))
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        self._build_net()
        self.loss = None
        self.cross_entropy = None
        self.train_op = None
        self.sigmoid_out = None
        self.test_eval = None
        self.feed_dict = None

    def _weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=1.0, name='weights')
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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

    def _build_net(self):
        self.y_conv = self._cnn_network()

    def _cnn_network(self):
        x_image = tf.reshape(self.x_pl, [-1,
                                         self.input_dim,
                                         self.input_timestep,
                                         16])
        W_conv1 = self._weight_variable([5, 5, 16, 32])
        b_conv1 = self._bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self._weight_variable([5, 5, 32, 64])
        b_conv2 = self._bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        #pdb.set_trace()

        W_fc1 = self._weight_variable([75 * 75 * 64, 384])
        b_fc1 = self._bias_variable([384])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 75*75*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self._weight_variable([384, 1])
        b_fc2 = self._bias_variable([1])
        #pdb.set_trace()

        return tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    def setup_loss_and_trainOp(self, learning_rate):
        # Calculate the loss
        self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                    self.y_conv, self.y_pl, self.pos_weight)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def do_eval(self, X_test, y_test):
        pass

    def do_train(
            self,
            ds,
            X_train,
            y_train,
            X_val,
            y_val,
            FLAGS):

        # Add the op to optimize
        self.setup_loss_and_trainOp(FLAGS.learning_rate)
        # Put there somewhere
        self.sigmoid_out = tf.nn.sigmoid(self.y_conv)
        output = tf.round(self.sigmoid_out)
        correct_prediction = tf.equal(self.y_pl, output)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Add the variable initializer op.
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print('Total number of 1 in validation set ', np.sum(y_val))
        for epoch in xrange(FLAGS.epochs):
            total_batch = len(X_train)/ds.batch_size
            for j in range(int(total_batch)):
                batch_xs, batch_ys = ds.next_training_batch(
                                        X_train,
                                        y_train)
                #pdb.set_trace()
                batch_ys = np.reshape(batch_ys ,(self.batch_size, 1))
                self.sess.run(self.train_op,
                              feed_dict={
                                self.x_pl: batch_xs,
                                self.y_pl: batch_ys,
                                self.keep_prob: 0.5})
            y_val = np.reshape(y_val ,(len(y_val),1))
            X_val = np.reshape(X_val ,(len(X_val),
                                       self.input_dim,
                                       self.input_timestep,
                                       16))
            # pdb.set_trace()
            test_accuracy = self.sess.run(accuracy,
                                          feed_dict={
                                            self.x_pl: X_val,
                                            self.y_pl: y_val,
                                            self.keep_prob: 1.0})
            if epoch == 0:
                best_accuracy = test_accuracy
                self.saver.save(self.sess,
                                (FLAGS.model_dir + FLAGS.train_set + ".ckpt"))
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.saver.save(self.sess, (FLAGS.model_dir + FLAGS.train_set + ".ckpt"))
                pred = self.sess.run(self.sigmoid_out,
                                     feed_dict={
                                        self.x_pl: X_val,
                                        self.keep_prob: 1.0})
                print('ROC curve: ', roc_auc_score(y_val, pred))
            print("step %d, test accuracy %g" % (epoch, test_accuracy))

        print("best accuracy %g" % (best_accuracy))

    def predict(self, ds, X_test, y_test, FLAGS):

        # Load the data and run model on test data
        print('Testing')
        print('---------------------------------------------------------------')
        X_test, ids = ds.load_test_data(FLAGS.test_set)
        # Restore model weights from previously saved model
        self.saver.restore(
            self.sess,
             (FLAGS.model_dir + FLAGS.train_set + ".ckpt"))
        predictions = []
        for i in range(len(X_test)):
            test_x = X_test[i]
            # pdb.set_trace()
            test_x = test_x.flatten()
            test_x_tensor = test_x.reshape(1,  test_x.shape[0])
            dic = {self.x_pl: test_x_tensor, self.keep_prob: 1.0}
            pred = self.sess.run(self.sigmoid_out, feed_dict=dic)
            predictions.append(pred)
        # Save the results
        frame = pd.DataFrame({'File': ids,
                              'Class': predictions
                              })
        cols = frame.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        frame = frame[cols]
        frame['Class'] = frame['Class'].astype(float)
        frame.to_csv(FLAGS.test_set + '_res.csv', index=False)
        print('Saved results in: ', FLAGS.test_set)
