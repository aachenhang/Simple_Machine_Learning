# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:34:05 2018

@author: Hang Chen
"""


import numpy as np


class SVC(object):
    def __init__(self, C = 1, kernel = 'linear', degree = 2, gamma = 1, coef0 = 0, tol = 1e-3, max_iter = 100):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
        assert kernel in kernel_list,  'kernel most be one of ' + str(kernel_list)
        if(kernel is 'linear'):
            self.kernel_func = self.linear_func
        elif(kernel is 'rbf'):
            self.kernel_func = self.rbf_func
        elif(kernel is 'poly'):
            self.kernel_func = self.poly_func
        elif(kernel is 'sigmoid'):
            self.kernel_func = self.sigmoid_func
    
    def linear_func(self, X, Y):
        return np.dot(X, Y.T)
    
    def rbf_func(self, X, Y):
        return np.exp(-self.gamma * np.sum((X-Y)**2, axis=1))
    
    def poly_func(self, X, Y):
        return (self.coef0 + np.dot(X, Y.T)) ** self.degree
    
    def sigmoid_func(self, X, Y):
        return np.tanh(self.coef0 + self.gamma * np.dot(X, Y.T))
    

    def find_bound(self, alpha_1, alpha_2, y_1, y_2):
        L, H = 0, 0
        if(y_1 == y_2):
            H = min(self.C, alpha_1 + alpha_2)
            L = max(0, alpha_1 + alpha_2 - self.C)
        else:
            H = min(self.C, self.C + alpha_2 - alpha_1)
            L = max(0, alpha_2 - alpha_1)
        return L, H

    def pick_random_except_i(self, n, i):
        num = i
        while(num == i):
            num = np.random.randint(low = 0, high=n)
        return num

    def fit(self, X, Y):
        self.intercept_ = 0
        self.alpha = np.zeros(shape=X.shape[0])
        self.support_ = []
        self.n_support_ = [0]
        self.support_vectors_ = []
        self.support_vectors_y = []
#        self.label = []
        
        assert isinstance(X, np.ndarray), "X must be class <numpy.ndarray>"
        assert isinstance(Y, np.ndarray), "Y must be class <numpy.ndarray>"
        assert X.shape[0] == Y.shape[0], "X.shape[0] must be euqal to Y.shape[0]"
        assert X.shape[0] > 2, "X.shape[0] must greater than 2"
        
        iter_ = 0
        n = X.shape[0]
        
        while iter_ < self.max_iter:
            iter_ += 1
            for i in range(n):
                K_i = self.kernel_func(X, X[i])
                u_i = np.dot(self.alpha*Y, K_i) + self.intercept_
                
                if Y[i]*u_i > 1 and self.alpha[i] > 0 \
                    or Y[i]*u_i == 1 and self.alpha[i]==0 \
                    or Y[i]*u_i == 1 and self.alpha[i]==self.C \
                    or Y[i]*u_i < 1 and self.alpha[i] < self.C:
                        j = self.pick_random_except_i(n, i)
                        E_i = u_i - Y[i]
                        K_j = self.kernel_func(X, X[j])
                        u_j = np.dot(self.alpha*Y, K_j) + self.intercept_
                        E_j = u_j - Y[j]
                        eta = K_i[i] + K_j[j] - 2 * K_i[j]
                        alpha_old = self.alpha.copy()
                        
                        L, H = self.find_bound(alpha_old[i], alpha_old[j], Y[i], Y[j])
                        
                        self.alpha[j] = alpha_old[j] + Y[j]*(E_i-E_j)/eta
                        self.alpha[j] = np.clip(self.alpha[j], L, H)
                        self.alpha[i] = alpha_old[i] + Y[j]/Y[i]*(alpha_old[j] - self.alpha[j])
                        
                        b_1 = self.intercept_ - E_i - Y[i] * (self.alpha[i] - alpha_old[i]) * K_i[i] - \
                                Y[2]*(self.alpha[j] - alpha_old[j]) * K_i[j]
                        b_2 = self.intercept_ - E_j - Y[i] * (self.alpha[i] - alpha_old[i]) * K_i[j] - \
                                Y[2]*(self.alpha[j] - alpha_old[j]) * K_j[j]
                                
                        if 0 < self.alpha[i] and self.alpha[i] < self.C:
                            self.intercept_ = b_1
                        elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                            self.intercept_ = b_2
                        else:
                            self.intercept_ = (b_1 + b_2) / 2
                        
#            if(np.sum(alpha_old - self.alpha)**2 < self.tol):
#                break

        if(self.kernel is 'linear'):
            self.coef = np.dot(self.alpha*Y, X)
        for i in range(n):
            if self.alpha[i] > 1e-9:
                self.n_support_[0] += 1
                self.support_vectors_.append(X[i])
                self.support_vectors_y.append(Y[i])
                self.support_.append(i)
        self.support_vectors_ = np.array(self.support_vectors_)
        self.support_vectors_y = np.array(self.support_vectors_y)
        self.support_ = np.array(self.support_)
        return
    
    @property
    def coef_(self):
        if(self.kernel is not 'linear'):
            raise AttributeError('coef_ is only available when using a linear kernal')
        else:
            return self.coef
    
    def predict(self, X):
        if(self.kernel is 'linear'):
            pred = np.sign(np.dot(self.coef_, X.T) + self.intercept_)
        else:
            s = 0
            for i in range(self.n_support_[0]):
                s += self.alpha[self.support_[i]] * self.support_vectors_y[i] * \
                    self.kernel_func(self.support_vectors_[i], X)
            pred = np.sign(s + self.intercept_)
        return pred

