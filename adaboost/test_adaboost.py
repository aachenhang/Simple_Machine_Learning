# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:29:00 2018

@author: Hang Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import sys  
from Adaboost import Adaboost



if __name__ == "__main__":
    assert float(sys.version.split(' ')[0].split('.')[0]) >= 3, "please run in python3"
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [3, 2]]
    Y = np.array([-1] * 20 + [1] * 20)
    adaboost = AdaBoost()
    adaboost.fit(X, Y)
    
    plt.figure(0, figsize=(4, 3))
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

#    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = adaboost.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(0, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
