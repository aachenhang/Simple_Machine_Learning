# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    a = 1.716
    b = 2.0 / 3
    return a * np.tanh(b*x)



def sigmoid_gradient(x):
    a = 1.716
    b = 2.0 / 3
    return 4 * a * b * np.exp(2 * b * x) / (np.exp(2 * b * x) + 1) ** 2



def mlp_inference(inputdata, w_1, w_2, network_structure, activate_func):
    inputlayer = inputdata.reshape([-1, network_structure[0]])
    inputlayer_add_ones = np.concatenate((inputlayer, np.ones(shape=[inputlayer.shape[0], 1])), axis=1)
    hidden_layer = np.dot(inputlayer_add_ones, w_1)
    hidden_layer_activate = sigmoid(hidden_layer)
    hidden_layer_add_ones = np.concatenate((hidden_layer_activate, np.ones(shape=[hidden_layer_activate.shape[0], 1])), axis=1)
    output_layer = np.dot(hidden_layer_add_ones, w_2)
    output_layer_activate = sigmoid(output_layer)
    return output_layer_activate


def mlp_train(inputdata, w_1, w_2, network_structure, label, learing_rate):
    inputlayer = inputdata.reshape([-1, network_structure[0]])
    inputlayer_add_ones = np.concatenate((inputlayer, np.ones(shape=[inputlayer.shape[0], 1])), axis=1)
    hidden_layer = np.dot(inputlayer_add_ones, w_1)
    hidden_layer_activate = sigmoid(hidden_layer)
    hidden_layer_add_ones = np.concatenate((hidden_layer_activate, np.ones(shape=[hidden_layer_activate.shape[0], 1])), axis=1)
    output_layer = np.dot(hidden_layer_add_ones, w_2)
    output_layer_activate = sigmoid(output_layer)
    
    loss = np.mean(1 / 2 * (output_layer_activate - label) ** 2)
    
    delta_outputlayer = (output_layer_activate - label) * sigmoid_gradient(output_layer)
    new_w_2 = w_2 - learing_rate * np.dot(delta_outputlayer, hidden_layer_add_ones).T
    
    delta_hiddenlayer = delta_outputlayer * w_2[:-1, :]
    new_w_1 = w_1 - learing_rate * np.dot(delta_hiddenlayer, inputlayer_add_ones).T
    
    hidden_layer = np.dot(inputlayer_add_ones, new_w_1)
    hidden_layer_activate = sigmoid(hidden_layer)
    hidden_layer_add_ones = np.concatenate((hidden_layer_activate, np.ones(shape=[hidden_layer_activate.shape[0], 1])), axis=1)
    output_layer = np.dot(hidden_layer_add_ones, new_w_2)
    output_layer_activate = sigmoid(output_layer)
    
    new_loss = np.mean(1 / 2 * (output_layer_activate - label) ** 2)
    return new_w_1, new_w_2, loss, new_loss


def drawMesh(title, output, xx, yy):
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, np.sign(output), cmap='binary')
    plt.title(title)
    

def stochastic_backpropagation(data, criteria, w_1, w_2, ground_truth):
    np.random.seed(seed=1234)
    x = [0]
    y = [1]
    step = 0.0
    error = 1
    epoch_size = data.shape[0]
    diff = 9999
    while(diff > criteria):
        sample_idx = np.random.randint(epoch_size)
        label = int(ground_truth[sample_idx, 0])
        w_1, w_2, loss, new_loss = mlp_train(inputdata=data[sample_idx], w_1=w_1, w_2=w_2,
                                    network_structure=[3,1,1],
                                    label=np.array([[label]]),
                                    learing_rate=0.1)
        diff = loss-new_loss
        output = mlp_inference(inputdata=data, w_1=w_1, w_2=w_2,
                           network_structure=[3,1,1],
                           activate_func=sigmoid)
        predict = np.sign(output)
        precision = np.mean(np.equal(predict, ground_truth))
        error = 1-precision
        step += 1
        x.append(step)
        y.append(error)
#        print("step %d, error %f, loss %f, diff %f" % (step , error, loss, diff))
    print(w_1, w_2)
    return x[:], y[:]
    

def drawPlot(title, x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title(title)
