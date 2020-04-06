import numpy as np
from utils import adam

"""
solve linear regression problem:
        Y = W @ X
        
"""

def ridge_regressor(X, Y, beta=1e-6):
    W = Y.dot(X.T.dot(np.linalg.inv(X.dot(X.T) + beta * np.eye(X.shape[0]))))
    return W

def spectral_norm(w, vector=False, max_iter = 10):
    v = np.random.random((w.shape[1],1))
    v /= np.linalg.norm(v)
    for i in range(max_iter):
        u = w @ v
        v = w.T @ u
        sig = np.linalg.norm(v)/np.linalg.norm(u)
        u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    if vector:
        return sig, u, v
    else:
        return sig

def gradient_descent(X, Y, slice):
    class MyLoss:
        def __init__(self, x, y, slice):
            self.x = x
            self.y = y
            self.l2 = 1e-6
            self.ls = 1
            self.slice = slice

        def value(self, w):
            return 0.5 * np.sum(np.square(self.y - w @ self.x)) + self.l2 * np.trace(w @ w.T)

        def gradient(self, w, itr):
            paddings = np.zeros(w.shape)
            if len(self.slice) > 0:
                s, u, v = spectral_norm(w[:,self.slice], vector=True)
                print('spectral radius:',s)
                paddings[:,self.slice] = s * u @ v.T
            return w @ self.x @ self.x.T - self.y @ self.x.T + 2 * self.l2 * w + self.ls * paddings

    loss = MyLoss(X, Y, slice)
    w0 = np.random.random( (Y.shape[0], X.shape[0]) )
    w = adam(loss, w0, num_iters=3000, step_size=0.1, b1=0.9, b2=0.999)
    return w