# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:29:00 2018

@author: Hang Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import sys  


class AdaBoost(object):
    def __init__(self):
        self.week_classifer_list = []
        return
        
    
    def classify(self, X, dimen, threshold, compare):
        pred = np.ones(shape=(X.shape[0], 1))
        if compare == 'lt':
            pred[X[:, dimen] <= threshold] = -1.0
        else:
            pred[X[:, dimen] > threshold] = 1.0
        return pred
    
    def get_week_classifer(self, X, Y, data_weight):
        assert isinstance(X, np.ndarray), "X must be class <numpy.ndarray>"
        assert isinstance(Y, np.ndarray), "Y must be class <numpy.ndarray>"
        assert X.shape[0] == Y.shape[0], "X.shape[0] must be euqal to Y.shape[0]"
        assert X.shape[0] > 2, "X.shape[0] must greater than 2"
        
        m, n = X.shape
        Y = np.array([Y])
        numSteps = 10
        week_classifer = dict()
        minErr = 9999999.0
        best_pred = np.zeros(shape=(m, 1))
        
        for i in range(n):
            rangeMin = X[:,i].min()
            rangeMax = X[:,i].max()
            stride = (rangeMax - rangeMin) / numSteps
            for j in range(-1, numSteps+1):
                for compare in ['lt', 'gt']:
                    threshold = (rangeMin + float(j) * stride)
                    pred = self.classify(X, i, threshold, compare)
                    errArr = np.ones(shape=(m,1)) 
                    errArr[pred==Y.T] = 0
                    sumErr = np.dot(data_weight.T , errArr)
                    
#                    if sumErr.shape != (0,0):
#                        print(data_weight)
#                        print(errArr)
#                        print(sumErr)
                    if sumErr[0,0] < minErr:
                        minErr = sumErr[0,0]
                        best_pred = pred.copy()
                        week_classifer['dimen'] = i
                        week_classifer['threshold'] = threshold
                        week_classifer['compare'] = compare
        return week_classifer, minErr, best_pred
        
        
    def fit(self, X, Y, max_iter = 40):
        m = X.shape[0]
        data_weight = np.ones(shape=(m, 1)) / m
        accum_pred = np.zeros(shape=(m, 1))
        for i in range(max_iter):
            week_classifer, err, pred = self.get_week_classifer(X, Y, data_weight)
            alpha = 0.5 * np.log((1.0-err) / max(err, 1e-16))
            week_classifer['alpha'] = alpha
            self.week_classifer_list.append(week_classifer)
            
            data_weight = data_weight * np.exp(np.multiply(-1*alpha*Y, pred))
            data_weight /= data_weight.sum()
            
            accum_pred += alpha * pred
            accum_err = np.multiply(np.sign(accum_pred) != Y.T, np.ones(shape=(m,1)))
            errorRate = accum_err.sum() / m
            if errorRate == 0.0:
                break
        
        return
        
        
    def predict(self, X):
        X = np.array(X)
        m = X.shape[0]
        accum_pred = np.zeros(shape=(m, 1))
        for i in range(len(self.week_classifer_list)):
            week_classifer = self.week_classifer_list[i]
            pred = self.classify(X, week_classifer['dimen'],
                            week_classifer['threshold'],
                            week_classifer['compare'])
            accum_pred += week_classifer['alpha'] * pred
        return np.sign(accum_pred)


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
        
        
        
        
        
        
        
        
        
        
        
        
        
        