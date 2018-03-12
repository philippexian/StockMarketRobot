"""
application of RNN with LSTM cells to MNIST dataset
each image is divided into 28 sequences, each contains 28 pixels

shape of tensor(batch_size, sequence-size, input_dim)

Hyperparameters:
input_dim 784 pixels
seq_size: number of sequences
hidden_dim: RNN size
epochs: time for the training process
batch_size: number of features one unit of time series carries
learning_rate
n_classes: from 0 to 9
keep_prob: the optional dropout is one of the common regularization technique. = 1- drop prob
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class RNN_LSTM:
    def __init__(self, input_dim, seq_size, n_classes=10, hidden_dim=128, learning_rate=0.01, batch_size=128, epochs=5, keep_prob=1.0):

        #Hyperparameters
        self.input_dim = input_dim
        self.seq_size=seq_size
        self.hidden_dim=hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size#??
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.keep_prob=keep_prob #should not be a constant

        #input and target placeholders
        X = tf.placeholder("float", [None, self.seq_size, self.input_dim])  # X training dataset
        Y = tf.placeholder("float")  # Y labels

        #weights and biases
        weights={'weights': tf.Variable(tf.random_normal([self.hidden_dim, self.n_classes]))}
        biases={'biases': tf.Variable(tf.random_normal([self.n_classes]))}#it's same in the whole process

        self.X=X
        self.Y = Y
        self.weights = weights
        self.biases = biases
        self.ouput = self.lstm()

        #loss function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.ouput, self.Y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)# calibrate to minimize the entropy with softmax regression

        self.correct = tf.equal(tf.arg_max(self.ouput, 1), tf.arg_max(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))  # mean of correct items

        # save the session
        self.saver = tf.train.Saver()

    def lstm(self):

        """
        prepare the input shape for the lstm, the oringinal shape is (batch_size, seq_size, input_dim)
        but must be transformed to a tensor list with the length seq_size, of the shape (batch_size, input_dim)

        must copy self.X to a new tensor X
        :return: output from the LSTM model
        """

        X=self.X
        #permute batch_size and seq_size
        X=tf.transpose(X, [1, 0, 2])#the [1] becomes [0], [0] becomes [1], [2] stays the same

        #reshape to (seq_size*batch_size, input_dim)
        X=tf.reshape(X, [-1, self.input_dim])

        #split the list of tensors
        X=tf.split(X, self.seq_size)

        #create lstm and add dropout
        lstm_cell=rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell=rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        outputs, states=rnn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

        output=tf.matmul(outputs[-1], self.weights['weights'])+self.biases['biases']#the last output and process
        return output

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())#first step in running a tf session

            for epoch in range(self.epochs):
                epoch_loss=0
                for i in range(int(data.train.num_examples/self.batch_size)):
                    batch_x, batch_y=data.train.next_batch(self.batch_size)

                    #set the correct shape of inputs
                    batch_x=batch_x.reshape((self.batch_size, self.seq_size, self.input_dim))
                    i, c=sess.run([self.optimizer, self.cost], feed_dict={self.X:batch_x, self.Y:batch_y})# run the functions in the first argument fed by the data in the second argument
                    epoch_loss+=c# cost function = the sum of loss of all batches
                print('Epoch ', (epoch+1), 'completed out of ', self.epochs, 'loss: ', epoch_loss)
            save_path=self.saver.save(sess, './lstm.ckpt')#save process parameters...
            print('Accuracy ', self.accuracy.eval({self.X: data.train.images.reshape((-1, self.seq_size, self.input_dim)), self.Y: data.train.labels}))


    def test(self, data):

        with tf.Session as sess:
            self.saver.restore(sess, './lstm.ckpt')
            print('Accuracy ', self.accuracy.eval({self.X: data.test.images.reshape((-1, self.seq_size, self.input_dim)), self.Y: data.test.labels}))

    def get_parameters(self):
        with tf.Session as sess:
            self.saver.restore(sess, './lstm.ckpt')
            weights, biases=sess.run([self.weights, self.biases])
            print('LSTM: \n',
                'weights: \n', weights['weights'], '\n',
                'biases: \n', biases['biases'], '\n')
            return weights, biases

