"""
RNN composed of LSTM cells
The input data are in a tensor of shape(batch size, sequence size, input_dim)
Hyperparameters:
input_dim
seq_size
batch_size
hidden_dim
epochs
learning_rate
keep_prob
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt

class RNN_LSTM:
    def __init__(self, input_dim, seq_size, hidden_dim=128, epochs=200, batch_size=128, learning_rate=0.01, keep_prob=1.0):
        #hyperparameters
        self.input_dim = input_dim
        self.seq_size=seq_size
        self.hidden_dim=hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob=keep_prob

        # input and target placeholders
        X = tf.placeholder("float", [None, seq_size, self.input_dim])  # X training dataset, the second parameter is the shape, the goal here is to keep the original shape
        Y = tf.placeholder("float", [None, 1])  # Y labels

        # weights and biases
        weights = tf.Variable(tf.random_normal([self.hidden_dim, 1]), name='weights')
        biases = tf.Variable(tf.random_normal([1]), name='b')

        self.X = X
        self.Y = Y
        self.weights = weights
        self.biases = biases
        self.ouput = self.lstm()

        # loss function and optimizer
        self.cost = tf.nn.l2_loss(self.ouput-self.Y)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)  # calibrate to minimize the l2 loss function

        self.accuracy = 1. - tf.reduce_mean(tf.abs((self.decoded - self.Y) / self.Y))  # sum of relative error

        # save the session
        self.saver = tf.train.Saver()


    def lstm(self):

        """
        prepare the input shape for the lstm, the oringinal shape is (batch_size, seq_size, input_dim)
        but must be transformed to a tensor list with the length seq_size, of the shape (batch_size, input_dim)

        must copy self.X to a new tensor X
        """

        X=self.X
        #permute batch_size and seq_size
        X=tf.transpose(X, [1, 0, 2])#the [1] becomes [0], [0] becomes [1], [2] stays the same

        #reshape to (seq_size*batch_size, input_dim)
        X=tf.reshape(X, [-1, self.input_dim])

        #split the list of tensors
        X=tf.split(X, self.seq_size)

        #create lstm and add dropout
        lstm_cell=rnn_cell.LSTMCell(self.hidden_dim, use_peepholes=True)
        lstm_cell=rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        outputs, states=rnn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

        output=tf.matmul(outputs[-1], self.weights)+self.biases#the last output and process
        return output

    def train(self, data, target):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())#first step in running a tf session

            for epoch in range(self.epochs):
                epoch_loss=0
                for i in range(int(len(data)/self.batch_size)):
                    start=i*self.batch_size
                    end=(i+1)*self.batch_size

                    batch_x = np.array(data[start:end])
                    batch_y = np.array(target[start:end])

                    batch_x = batch_x.reshape(batch_x, (self.batch_size, self.seq_size, self.input_dim))
                    batch_y = batch_y.reshape(batch_y, (self.batch_size, 1))

                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_x, self.Y: batch_y})  # run the functions in the first argument fed by the data in the second argument
                    epoch_loss += c  # cost function of this epoch = the sum of loss of all batches
                print('Epoch ', (epoch+1), 'completed out of ', self.epochs, 'loss: ', epoch_loss)
            save_path=self.saver.save(sess, './lstm.ckpt')#save process parameters
            print('Accuracy ', self.accuracy.eval({self.X: data.reshape((-1, self.seq_size, self.input_dim)), self.Y: target.reshape(-1, 1)}))


    def test(self, data, target):#here the data should be the test set
        with tf.Session as sess:
            self.saver.restore(sess, './lstm.ckpt')

            print('Accuracy ', self.accuracy.eval({self.X: data.reshape((-1, self.seq_size, self.input_dim)), self.Y: target.reshape(-1, 1)}))

            prediction=sess.run(self.ouput, feed_dict={self.X: data.reshape((-1, self.seq_size, self.input_dim))})
            plt.plot(prediction, linewidth=0.3, color='r', label='predicted values')
            plt.plot(target, linewidth=0.3, color='r', label='target values')
            plt.legend()
            plt.show()
            



