import numpy as np

class PCA(object):
    def __init__(self, n_components = None):
        assert n_components is not None, "n_components cannot be None"
        self.n_components = n_components
        return
    
    def svd_flip(self, u, v, u_based_decision=True):
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, np.arange(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[np.arange(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v
    
    def fit(self, X):
        assert isinstance(X, np.ndarray), "X must be class <numpy.ndarray>"
        
        X = X - np.mean(X, axis=0)
        self.U, self.S, self.V = np.linalg.svd(X)
#        self.U, selfV = self.svd_flip(self.U, self.V)
        idx = np.argsort(self.S)
        self.n_components = min(idx.shape[0], self.n_components)
        idx = idx[self.n_components-1::-1]
        self.remain_vec = self.V[:, idx]
        return self
    
    def transform(self, X):
        assert isinstance(X, np.ndarray), "X must be class <numpy.ndarray>"
        assert X.shape[1] == self.remain_vec.shape[0], \
            "X.shape[0] is not equal to remain_vec.shape[0]"
        self.U = self.U[:, :self.n_components]

#        return np.dot(X, self.remain_vec)
        return self.U[:, :self.n_components]* self.S[:self.n_components]
        self.U = self.U[:, :self.n_components]

        self.U *= self.S[:self.n_components]

        return self.U