"""
a deep autoencoder system consisting of encoder and decoder, each has 3 layers
inout: a tensor (batch_size, time-series size) assuming only one feature is tested here. otherwise (batch_size, time interval size, num_features)
the goal after the encoding is to represent a financial scenario

Hyperparameters:
- input_dim: = time series size
- epochs: time for training
- batch_size: 128 here
- learning_rate
- n_examples: number of instances to visualize the input and the encoding result
"""

import tensorflow as tf
import numpy as np

class Deep_Autoencoder:
    def __init__(self, input_dim, n_nodes_hl=(32, 16, 1), epoch=400, batch_size=128, learning_rate=0.01, n_examples=10):

        #hyperparameters
        self.input_dim=input_dim
        self_n_nodes_hl=n_nodes_hl
        self.epoch=epoch
        self.n=batch_size=batch_size
        self.learning=learning_rate
        self.n_examples=n_examples

        # input and target placeholders
        X = tf.placeholder("float", [None, self.seq_size, self.input_dim])  # X training dataset
        Y = tf.placeholder("float", [None, self.seq_size, self.input_dim])  # the output is still of the input dimension

        'define the bottleneck structure' \
        'random initialization' \
        'use sigmoid function as activation function except the last layer of decoder'

        with tf.name_scope('encoder'):
            weights_e = {
                'encoder_hl1': tf.Variable(tf.random_normal([self.input_dim, n_nodes_hl[0]]), name='weights_e1'),
                'encoder_hl2': tf.Variable(tf.random_normal([n_nodes_hl[0], n_nodes_hl[1]]), name='weights_e2'),
                'encoder_hl3': tf.Variable(tf.random_normal([n_nodes_hl[1], n_nodes_hl[2]]), name='weights_e3')
            }
            biases_e = {
                'b_hl1': tf.Variable(tf.random_normal([n_nodes_hl[0]]), name='biases_e1'),
                'b_hl2': tf.Variable(tf.random_normal([n_nodes_hl[1]]), name='biases_e2'),
                'b_hl3': tf.Variable(tf.random_normal([n_nodes_hl[2]]), name='biases_e3')
            }

            layer_1 = tf.add(tf.matmul(X, weights_e['encoder_hl1']), biases_e['b_hl1'])  # matmul: multiple matrices in elementwise way
            layer_1 = tf.nn.sigmoid(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights_e['encoder_hl2']),  biases_e['b_hl2'])
            layer_2 = tf.nn.sigmoid(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights_e['encoder_hl3']), biases_e['b_hl3'])
            layer_3 = tf.nn.sigmoid(layer_3)


        with tf.name_scope('decoder'):
            weights_d = {
                'decoder_hl1': tf.Variable(tf.random_normal([n_nodes_hl[2], n_nodes_hl[1]]), name='weights_e1'),
                'decoder_hl2': tf.Variable(tf.random_normal([n_nodes_hl[1], n_nodes_hl[0]]), name='weights_e2'),
                'decoder_hl3': tf.Variable(tf.random_normal([n_nodes_hl[0], self.input_dim]), name='weights_e3')
            }
            biases_d = {
                'b_hl1': tf.Variable(tf.random_normal([n_nodes_hl[1]]), name='biases_e1'),
                'b_hl2': tf.Variable(tf.random_normal([n_nodes_hl[0]]), name='biases_e2'),
                'b_hl3': tf.Variable(tf.random_normal([self.input_dim]), name='biases_e3')
            }

            layer_4 = tf.add(tf.matmul(layer_3, weights_d['decoder_hl1']), biases_d['b_hl1'])
            layer_4 = tf.nn.sigmoid(layer_4)

            layer_5 = tf.add(tf.matmul(layer_4, weights_d['decoder_hl2']), biases_d['b_hl2'])
            layer_5 = tf.nn.sigmoid(layer_5)

            output_layer = tf.add(tf.matmul(layer_5, weights_d['decoder_hl3']), biases_d['b_hl3'])

            self.X=X
            self.Y=Y
            self.weights={'weights_e': weights_e, 'weights_d': weights_d}
            self.biases={'biases_e':biases_e, 'biases_d':biases_d}
            self.encoded=layer_3
            self.decoded=output_layer

            #loss function and optimizer
            self.cost = tf.nn.l2_loss(self.decoded-self.Y)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            self.accuracy = 1.-tf.reduce_mean(tf.abs((self.decoded-self.Y)/self.Y))#sum of relative error

            # save the session
            self.saver = tf.train.Saver()


    def train_neural_network(self, data, targets):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.epoch):
                epoch_loss=0
                i=0

                #input to the LSTM
                hidden_vec=np.array([])
                #training in batch
                while i<len(data):
                    start=i
                    end=i+self.batch_size

                    batch_x=np.array(data[start:end])
                    batch_y=np.array(targets[start:end])

                    # run the functions in the first argument fed by the data in the second argument
                    hidden, _, c = sess.run([self.encoded, self.optimizer, self.cost], feed_dict={self.X: batch_x, self.Y: batch_y})# loops and substitutes the values in 'feed_dict'
                    epoch_loss+=c
                    hidden_vec=np.append(hidden_vec, hidden)#hidden: the fragment of hidden vector
                    i+=self.batch_size

                if (epoch+1)%50==0:
                    print('Epoch', epoch+1, 'completed out of ', self.epochs, 'loss: ', epoch_loss)

            self.saver.save(sess, './Deep_Autoencoder.ckpt')
            print('Accuracy ', self.accuracy.eval({self.X: data, self.Y: targets}))

        return hidden_vec

    def test_neural_network(self, data, targets):
        with tf.Session() as sess:
            #invoke the saved session
            self.saver.restore(sess, './Deep_Autoencoder.ckpt')
            hidden_vec = np.array([])
            i=0

            while i < len(data):
                start = i
                end = i + self.batch_size

                batch_x = np.array(data[start:end])

                hidden = sess.run(self.encoded, feed_dict={self.X: batch_x})  # loops and substitutes the values in 'feed_dict'
                hidden_vec = np.append(hidden_vec, hidden)  # hidden: the fragment of hidden vector
                i += self.batch_size

            #why we need hidden vector??
            reconstruction=sess.run(self.decoded, feed_dict={self.X: data})

            #show some details
            for i in range(self.n_examples):
                print('input: ', data[i], '\t', 'compressed: ', hidden_vec[i], '\t', 'output: ', reconstruction[i], '\n')

        #evaluate accuracy
        print('Accuracy ', self.accuracy.eval({self.X: data, self.Y: targets}))
        return hidden_vec

    def get_parameters(self):
        with tf.Session as sess:
            self.saver.restore(sess, './Deep_Autoencoder.ckpt')
            weights, biases=sess.run([self.weights, self.biases])
            print('Encoder: \n',
                'Hidden layer 1: \n',
                  'weights: \n', weights['weights_e']['encoder_hl1'], '\n',
                  'biases: \n', biases['biases_e']['b_hl1'], '\n'
                  'Hidden layer 2: \n',
                  'weights: \n', weights['weights_e']['encoder_hl2'], '\n',
                  'biases: \n', biases['biases_e']['b_hl2'], '\n'
                  'Hidden layer 3: \n',
                  'weights: \n', weights['weights_e']['encoder_hl3'], '\n',
                  'biases: \n', biases['biases_e']['b_hl3'], '\n',
                  'Decoder: \n',
                'Hidden layer 1: \n',
                  'weights: \n', weights['weights_d']['decoder_hl1'], '\n',
                  'biases: \n', biases['biases_d']['b_hl1'], '\n'
                  'Hidden layer 2: \n',
                  'weights: \n', weights['weights_d']['decoder_hl2'], '\n',
                  'biases: \n', biases['biases_d']['b_hl2'], '\n'
                  'Hidden layer 3: \n',
                  'weights: \n', weights['weights_d']['decoder_hl3'], '\n',
                  'biases: \n', biases['biases_d']['b_hl3'], '\n'
                  )
            return weights, biases

