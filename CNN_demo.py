"""
Application of CNN on the MNIST dataset
The model is composed with 2 convolutional layers, one fully connected layer and an output layer.
each image is flattened to a 28*28*3 vector, stored in the tensor X

Hyperparameters:
input_dim 784 pixels * 3
epochs: time for the training process
batch_size: use"batch training", the batch size is usually 64, 128...
learning_rate
n_classes: from 0 to 9
"""

import tensorflow as tf
import numpy as np

class Convolutional_Net:
    def __init__(self, input_dim, n_nodes_hl, epochs=10, batch_size=128, learning_rate=0.01, n_classes=10):

        #hyperparameters
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        #input and target placeholders

        X=tf.placeholder("float", [None, self.input_dim])# X training dataset
        Y=tf.placeholder("float")#Y labels

        with tf.name_scope('CNN'):

            #as usual, weights and biases are in form of a dictionary
            #first convolutional layer: TODO 5*5 patch size ??
            #it takes just one input and returns n_nodes_hl[0] outputs

            #second layer is the same

            #densely connected layer: it is added once the image size is reduced to 7*7 (after the first 2 convolutional layers) TODO what it is for??

            weights={
                'weights_convl1': tf.Variable(tf.random_normal([5, 5, 1, n_nodes_hl[0]])),
                'weights_convl2': tf.Variable(tf.random_normal([5, 5, n_nodes_hl[0], n_nodes_hl[1]])),
                'weights_fully_connected': tf.Variable(tf.random_normal([7*7*n_nodes_hl[1], n_nodes_hl[2]])),
                'weights_out': tf.Variable(tf.random_normal([n_nodes_hl[2], self.n_classes]))
            }
            biases={
                'biases_convl1': tf.Variable(tf.random_normal([n_nodes_hl[0]])),
                'biases_convl2': tf.Variable(tf.random_normal([n_nodes_hl[1]])),
                'biases_fully_connected': tf.Variable(tf.random_normal([n_nodes_hl[2]])),
                'biases_out': tf.Variable(tf.random_normal([self.n_classes]))
            }

            #reshape the image set to 4d tensor: (number_of_images, width, height, color channels)
            X=tf.reshape(X, shape=[-1, 28, 28, 1])#black and white only

            #tf.nn.conv2d: computes a 2d convolution given a 4d tensor through a sliding window(stride of 1 pixel for each dimension)
            # apply the rectified linear activation function on the computed output
            #tf.nn.max_pool: performs max pooling over 2*2 blocks. It halves the image size to 14*14
            # computes the output of the densely connected layer and apply again the rectified activation function, layer layer is the processing of the entire image

            conv1=tf.nn.relu(tf.nn.conv2d(X, weights['weights_convl1'], strides=[1, 1, 1, 1], padding='SAME')+biases['biases_convl1'])
            conv1=tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# each layer contains a convolutional processing and a pooling

            conv2=tf.nn.relu(tf.nn.conv2d(conv1, weights['weights_convl2'], strides=[1, 1, 1, 1], padding='SAME') + biases['biases_convl2'])
            conv2=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            fully_connected=tf.reshape(conv2, [-1, 7*7*n_nodes_hl[1]])#reduce the image size to 7*7
            fully_connected=tf.nn.relu(tf.matmul(fully_connected, weights['weights_fully_connected'])+biases['biases_fully_connected'])

            output=tf.matmul(fully_connected, weights['weights_out'])+biases['biases_out']

            self.X = X
            self.Y = Y
            self.weights = weights
            self.biases = biases
            self.ouput = output

            #loss function and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.ouput, self.Y))  # calibrate to minimize the entropy with softmax regression
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            self.correct = tf.equal(tf.arg_max(self.ouput, 1), tf.arg_max(Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))  # mean of correct items

            # save the session
            self.saver = tf.train.Saver()


    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())#first step in running a tf session

            for epoch in range(self.epochs):
                epoch_loss=0
                for i in range(int(data.train.num_examples/self.batch_size)):
                    x, y=data.train.next_batch(self.batch_size)

                    #set the correct shape of inputs
                    x=x.reshape((self.batch_size, 28, 28, 1))
                    i, c=sess.run([self.optimizer, self.cost], feed_dict={self.X:x, self.Y:y})# run the functions in the first argument fed by the data in the second argument
                    epoch_loss+=c# cost function = the sum of loss of all batches
                print('Epoch ', (epoch+1), 'completed out of ', self.epochs, 'loss: ', epoch_loss)
            self.saver.save(sess, './ConvNet.ckpt')#save process parameters...
            print('Accuracy ', self.accuracy.eval({self.X: x.reshape((-1, 28, 28, 1)), self.Y: y}))

    def test(self, data):
        x, y = data.test.nexy_batch(1500)

        with tf.Session as sess:
            self.saver.restore(sess, './ConvNet.ckpt')
            print('Accuracy ', self.accuracy.eval({self.X: x.reshape((-1, 28, 28, 1)), self.Y: y}))

    def get_parameters(self):
        with tf.Session as sess:
            self.saver.restore(sess, './ConvNet.ckpt')
            weights, biases=sess.run([self.weights, self.biases])
            print('Convolutional layer 1: \n',
                  'weights: \n', weights['weights_convl1'], '\n',
                  'biases: \n', biases['biases_convl1'], '\n'
                  'Convolutional layer 2: \n',
                  'weights: \n', weights['weights_convl2'], '\n',
                  'biases: \n', biases['biases_convl2'], '\n'
                  'Fully connected layer: \n',
                  'weights: \n', weights['weights_fully_connected'], '\n',
                  'biases: \n', biases['biases_fully_connected'], '\n',
                  'Output layer: \n',
                  'weights: \n', weights['weights_out'], '\n',
                  'biases: \n', biases['biases_out'], '\n',
                  )
            return weights, biases