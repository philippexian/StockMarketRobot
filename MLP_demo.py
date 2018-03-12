"""
Application of MLP to a MNIST dataset
the network is composed by 3 hidden layers, each of which consists of 500 units
each of the image is 28*28=784 pixels

Hyperparameters:
input_dim 784 pixels * 3
epochs: time for the training process
batch_size: use"batch training", the batch size is usually 64, 128...
learning_rate
n_classes: from 0 to 9
"""

import tensorflow as tf

class MLP:
    def __init__(self, input_dim, n_nodes_hl, epochs=10, batch_size=128, learning_rate=0.01, n_classes=10):
        self.input_dim=input_dim
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.n_classes=n_classes

        #like java initialization

        #input and target placeholders

        X=tf.placeholder("float", [None, self.input_dim])# X training dataset, the second parameter is the shape, the goal here is to keep the original shape, which is a row
        Y=tf.placeholder("float")#Y labels

        # weights and are of shape (number of units of the previous layer, number of units of this layer)
        # biases are of the shape (number of units of this layer, ) a vector

        with tf.name_scope('MLP'):
            weights={
                'weights_hl1': tf.Variable(tf.random_normal([self.input_dim, n_nodes_hl[0]])),
                'weights_hl2': tf.Variable(tf.random_normal([n_nodes_hl[0], n_nodes_hl[1]])),
                'weights_hl3': tf.Variable(tf.random_normal([n_nodes_hl[1], n_nodes_hl[2]])),
                'weights_out': tf.Variable(tf.random_normal([n_nodes_hl[2], self.n_classes]))
            }
            biases={
                'biases_hl1': tf.Variable(tf.random_normal([n_nodes_hl[0]])),
                'biases_hl2': tf.Variable(tf.random_normal([n_nodes_hl[1]])),
                'biases_hl3': tf.Variable(tf.random_normal([n_nodes_hl[2]])),
                'biases_out': tf.Variable(tf.random_normal([self.n_classes]))
            }

            output_layer_1=tf.add(tf.matmul(X, weights['weights_hl1']), biases['biases_hl1'])# matmul: multiple matrices in elementwise way
            output_layer_1=tf.nn.relu(output_layer_1)

            output_layer_2 = tf.add(tf.matmul(output_layer_1, weights['weights_hl2']), biases['biases_hl2'])  # matmul: multiple matrices in elementwise way
            output_layer_2 = tf.nn.relu(output_layer_2)

            output_layer_3 = tf.add(tf.matmul(output_layer_2, weights['weights_hl3']), biases['biases_hl3'])  # matmul: multiple matrices in elementwise way
            output_layer_3 = tf.nn.relu(output_layer_3)

            output=tf.matmul(output_layer_3, weights['weights_out'])+ biases['biases_out']

            self.X=X
            self.Y=Y
            self.weights=weights
            self.biases=biases
            self.ouput=output

            #loss function and optimizer
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.ouput, Y))# calibrate to minimize the entropy with softmax regression
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            self.correct=tf.equal(tf.arg_max(self.ouput, 1), tf.arg_max(Y, 1))
            self.accuracy=tf.reduce_mean(tf.cast(self.correct, 'float'))#mean of correct items

            #save the session
            self.saver=tf.train.Saver()

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())#first step in running a tf session

            for epoch in range(self.epochs):
                epoch_loss=0
                for i in range(int(data.train.num_examples/self.batch_size)):
                    x, y=data.train.next_batch(self.batch_size)
                    i, c=sess.run([self.optimizer, self.cost], feed_dict={self.X:x, self.Y:y})#run the functions in the first argument fed by the data in the second argument
                    epoch_loss+=c# cost function = the sum of loss of all batches
                print('Epoch ', (epoch+1), 'completed out of ', self.epochs, 'loss: ', epoch_loss)
            self.saver.save(sess, './MLP.ckpt')#save process parameters...
            print('Accuracy ', self.accuracy.eval({self.X: data.train.images, self.Y: data.train.labels}))

    def test(self, data):
        with tf.Session as sess:
            self.saver.restore(sess, './MLP.ckpt')
            print('Accuracy ', self.accuracy.eval({self.X: data.test.images, self.Y: data.test.labels}))

    def get_parameters(self):
        with tf.Session as sess:
            self.saver.restore(sess, './MLP.ckpt')
            weights, biases=sess.run([self.weights, self.biases])
            print('Hidden layer 1: \n',
                  'weights: \n', weights['weights_hl1'], '\n',
                  'biases: \n', biases['biases_hl1'], '\n'
                  'Hidden layer 2: \n',
                  'weights: \n', weights['weights_hl2'], '\n',
                  'biases: \n', biases['biases_hl2'], '\n'
                  'Hidden layer 3: \n',
                  'weights: \n', weights['weights_hl3'], '\n',
                  'biases: \n', biases['biases_hl3'], '\n',
                  'Output layer: \n',
                  'weights: \n', weights['weights_out'], '\n',
                  'biases: \n', biases['biases_out'], '\n',
                  )
            return weights, biases
