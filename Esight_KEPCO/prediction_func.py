##trainging set을 RNN을 통해 학습하고 예측
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.reset_default_graph()
tf.set_random_seed(777)

def prediction_rnn(data,colums,seq_length,hidden_dim,learning_rate = 0.01,iterations = 3001):
    #set parameters
    seq_length = seq_length
    data_dim =len(data[0])
    hidden_dim = hidden_dim
    output_dim = len(data[0])-colums
    learning_rate = learning_rate
    iterations = iterations
    train_size = int(len(data)*0.8)
    y = data_dim-output_dim
    data = MinMaxScaler(data)
    
    
    X = tf.placeholder(tf.float32,[None, seq_length,data_dim])
    Y = tf.placeholder(tf.float32,[None,output_dim])
    
    #divide data set to train and test set
    train_set = data[:train_size]
    test_set = data[train_size:]

    #make dataset for learning
    trainX, trainY = build_dataset(train_set,seq_length,y)
    testX,testY = build_dataset(test_set, seq_length,y)
    
    #make lstm cell
    Y_pred,loss,train = make_rnn_cell(X,Y,output_dim,hidden_dim,learning_rate)
    
    #learn, predict, accuracy
    df_pred = learn_prediction(X,Y,Y_pred,loss,train,iterations,trainX,trainY,testX)
    
    return df_pred

def prediction_lstm(data,colums,seq_length,hidden_dim,learning_rate = 0.01,iterations = 3001):
    #set parameters
    seq_length = seq_length
    data_dim =len(data[0])
    hidden_dim = hidden_dim
    output_dim = len(data[0])-colums
    learning_rate = learning_rate
    iterations = iterations
    train_size = int(len(data)*0.8)
    y = data_dim-output_dim
    data = MinMaxScaler(data)
    
    
    X = tf.placeholder(tf.float32,[None, seq_length,data_dim])
    Y = tf.placeholder(tf.float32,[None,output_dim])
    
    #divide data set to train and test set
    train_set = data[:train_size]
    test_set = data[train_size:]

    #make dataset for learning
    trainX, trainY = build_dataset(train_set,seq_length,y)
    testX,testY = build_dataset(test_set, seq_length,y)
    
    #make lstm cell
    Y_pred,loss,train = make_lstm_cell(X,Y,output_dim,hidden_dim,learning_rate)
    
    #learn, predict, accuracy
    df_pred = learn_prediction(X,Y,Y_pred,loss,train,iterations,trainX,trainY,testX)
    
    return df_pred

def prediction_gru(data,colums,seq_length,hidden_dim,learning_rate = 0.01,iterations = 3001):
    #set parameters
    seq_length = seq_length
    data_dim =len(data[0])
    hidden_dim = hidden_dim
    output_dim = len(data[0])-colums
    learning_rate = learning_rate
    iterations = iterations
    train_size = int(len(data)*0.7)
    validation_size = int(len(data)*0.2)
    y = data_dim-output_dim
    data = MinMaxScaler(data)
    
    X = tf.placeholder(tf.float32,[None, seq_length,data_dim])
    Y = tf.placeholder(tf.float32,[None,output_dim])
    
    #divide data set to train and test set
    train_set = data[:train_size]
    validation_set = data[train_size:train_size+validation_size]
    test_set = data[train_size+validation_size:]

    #make dataset for learning
    trainX, trainY = build_dataset(train_set,seq_length,y)
    testX,testY = build_dataset(test_set, seq_length,y)
    
    #make lstm cell
    Y_pred,loss,train = make_gru_cell(X,Y,output_dim,hidden_dim,learning_rate)
    
    #learn, predict, accuracy
    df_pred = learn_prediction(X,Y,Y_pred,loss,train,iterations,trainX,trainY,testX)
    
    return df_pred


def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0)-np.min(data,0)
    return numerator / (denominator+1e-7)

def data_max_min(data):
    data = np.array(data,np.float32)
    data_max = np.max(data,0)
    data_min = np.min(data,0)
    return data_max,data_min

def ReturnToOriginal(data,data_max,data_min):
    data_max = data_max-data_min+(1e-7)
    for i in range(0,len(data)):
        data[i,:] = (data[i,:]*data_max)+data_min
    return data

# build data
def build_dataset(time_series, seq_length, y):
    dataX = []
    dataY = []
    for i in range (0,len(time_series)-seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i+seq_length,y:]
     
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)


def make_rnn_cell(X,Y,output_dim,hidden_dim,learning_rate,activation_func = tf.tanh):
    with tf.variable_scope("rnn"):
        rnn_1 = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim,activation=activation_func),output_size=output_dim)
        rnn_2 = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim*2,activation=activation_func),output_size=output_dim)
        rnn_3 = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim,activation=activation_func),output_size=output_dim)
        cell = tf.contrib.rnn.MultiRNNCell([rnn_1,rnn_2,rnn_3])
        outputs,_states = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim,activation_fn = None)
        loss =tf.reduce_mean(tf.square(Y_pred-Y))
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    return Y_pred,loss,train


def make_lstm_cell(X,Y,output_dim,hidden_dim,learning_rate,activation_func = tf.tanh):
    with tf.variable_scope("lstm"):
        lstm_1 = tf.contrib.rnn.LSTMCell(num_units=hidden_dim, state_is_tuple = True, activation=activation_func)
        lstm_2 = tf.contrib.rnn.LSTMCell(num_units=hidden_dim*2, state_is_tuple = True, activation=activation_func)
        lstm_3 = tf.contrib.rnn.LSTMCell(num_units=hidden_dim, state_is_tuple = True, activation=activation_func)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_1,lstm_2,lstm_3])
        outputs,_states = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim,activation_fn = None)
        loss =tf.reduce_sum(tf.square(Y_pred-Y))
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    return Y_pred,loss,train


def make_gru_cell(X,Y,output_dim,hidden_dim,learning_rate):
    with tf.variable_scope("gru"):
        gru_1 = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
        gru_2 = tf.contrib.rnn.GRUCell(num_units=hidden_dim*2)
        gru_3 = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
        cell = tf.contrib.rnn.MultiRNNCell([gru_1,gru_2,gru_3])
        outputs,_states = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim,activation_fn = None)
        loss =tf.reduce_mean(tf.square(Y_pred-Y))
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    return Y_pred,loss,train

def learn_prediction(X,Y,Y_pred,loss,train,iterations,trainX,trainY,testX):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            _, step_loss = sess.run([train,loss],feed_dict={X:trainX, Y:trainY})
            if i%200 == 0:
                print("[step: {}] loss: {}".format(i,step_loss))
        
        test_predict = sess.run(Y_pred, feed_dict = {X:testX})
    
    return test_predict

def mae(data1,data2):
    return np.average(np.abs(data1-data2))

def accuracy(data1,data2):
    return np.sum(np.abs(data1-data2)<20)/(len(data1)*len(data1[0]))*100


# In[ ]:




