# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time# -*- coding: utf-8 -*-
from mlp import *


if __name__ == '__main__':
    data  = np.loadtxt('data.txt')

    data = data.reshape([10, 3, 3]) #size = [sample.size(), w.size(), x.size()]
    data = data[:, :-1, :]
    ground_truth = np.concatenate((-1 * np.ones(shape=[data.shape[0], 1]),
                                1 * np.ones(shape=[data.shape[0], 1])), axis=1).reshape([-1, 1])
    data = data.reshape([-1, 3])
    
    xx,yy = np.meshgrid(np.arange(-5,5,0.01),np.arange(-5,5,0.01))
    
    x1 = xx.reshape([1000, 1000, 1])
    
    x2 = yy.reshape([1000, 1000, 1])
    
    inputdata = np.concatenate((x1, x2), axis=2)


    w_1 = np.array([[0.5, -0.5], [0.3, -0.4], [-0.1, 1.0]])
    w_2 = np.array([[1.0], [-2.0], [0.5]])
    output = mlp_inference(inputdata=inputdata, w_1=w_1, w_2=w_2,
                               network_structure=[2, 2, 1], activate_func=sigmoid)
    drawMesh("input space", output.reshape([1000, 1000]), xx, yy)
    
    
    w_1 = np.array([[-1.0, 1.0], [-0.5, 1.5], [1.5, -0.5]])
    w_2 = np.array([[0.5], [-1.0], [1.0]])
    output = mlp_inference(inputdata=inputdata, w_1=w_1, w_2=w_2,
                               network_structure=[2, 2, 1], activate_func=sigmoid)
    drawMesh("input space", output.reshape([1000, 1000]), xx, yy)
    
    criteria = 0.000001
    np.random.seed(int(time.time()))
    w_1 = 2 * np.random.rand(4, 1) - 1
    w_2 = 2 * np.random.rand(2, 1) - 1
    x1, y1 = stochastic_backpropagation(data=data, criteria=criteria,
                                      w_1=w_1, w_2=w_2,
                                      ground_truth=ground_truth)
    drawPlot("random weight", x1, y1)
    
    w_1 = 0.5 * np.ones(shape=[4, 1])
    w_2 = -0.5 * np.ones(shape=[2, 1])
    x2, y2 = stochastic_backpropagation(data=data, criteria=criteria,
                                      w_1=w_1, w_2=w_2,
                                      ground_truth=ground_truth)
    drawPlot("same weight", x2, y2)
