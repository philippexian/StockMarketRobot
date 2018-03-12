"""
the data is transformed into a vector
shuffled_train_x, shuffled_train_y are used to train the autoencoder. Trained on a shuffled features set the algorithm reaches a higher accuracy
train_x, train_y are the ordered version of Train_x and Train_y, the algorithm is tested on them ??
test_x and test_y are test sets
"""

import csv
import numpy as np
import random

def sample_handling(fname):
    featureSet=[]

    with open(fname, 'r') as csvFile:
        readCSV=csv.reader(csvFile, delimiter=',')

        for row in readCSV:
            feature=list(row[1:6])#use 5 features as input: open, close, high, low and volume
            featureSet.append(feature)
        return featureSet

def create_feature_sets(fname, test_size=0.2):
    features=sample_handling(fname)
    features=np.array(features, dtype=np.float32)

    testing_size=int(test_size*len(features))
    train_x=list(features[:-testing_size])
    train_y=train_x#the result after decoding should be close to the original time-series

    shuffled_train_x=train_x.copy()
    random.shuffle(shuffled_train_x)
    shuffled_train_y=shuffled_train_x

    test_x=list(features[-testing_size:])
    test_y=test_x.copy()

    return train_x, train_y, shuffled_train_x, shuffled_train_y, test_x, test_y


"""
creates lstm dataset, given the original autoencoder train and test sets, taking in account the look-back
we predict the sequence at future times

for example, we predict the stock price S(t1+1) with the training set S(t0) to S(t1) 
"""

def create_lstm_dataset(train_x, train_y, test_x, test_y, look_back):
    train_X, train_Y, test_X, test_Y=[], [], [], []
    for i in range(len(train_x)-look_back-1):
        item_x=train_x[i:(i+look_back)]
        train_X.append(item_x)
        train_Y.append(train_y[i+look_back][0])#open price
    for i in range(len(test_x)-look_back-1):
        item_x_t=test_x[i:(i+look_back)]
        test_X.append(item_x_t)
        test_Y.append(test_y[i+look_back][0])
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

