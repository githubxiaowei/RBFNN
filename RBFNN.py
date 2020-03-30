import numpy as np
from scipy.cluster.vq import kmeans2
from utils import pairwise_dtw_distances


class RBFNN:
    def __init__(self, N_h, skip_con=True, bias=True, reweight=True):

        self.skip_con = skip_con
        self.bias = bias
        self.reweight = reweight
        self.N_i = None
        self.N_o = None
        self.N_h = N_h
        self.N_f = N_h
        self.sigma = 1
        self.beta = 1e-6

    @staticmethod
    def pairwise_distances(X, Y):
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return D

    @staticmethod
    def col_softmax(x):
        x_col_max = x.max(axis=0)
        x_col_max = x_col_max.reshape([1, x.shape[1]])
        x = x - x_col_max
        x_exp = np.exp(x)
        x_exp_col_sum = x_exp.sum(axis=0).reshape([1, x.shape[1]])
        softmax = x_exp / x_exp_col_sum
        return softmax

    @staticmethod
    def col_normalize(x):
        x_sum = x.sum(axis=0).reshape([1, x.shape[1]])
        softmax = x / x_sum
        return softmax

    def _pre_output(self, X):

        threshold = 1

        #         X -= np.min(X,axis=0)

        # 隐层输出
        H = np.exp(-self.pairwise_distances(self.W_i, X.T) / threshold)

        if self.skip_con:
            H = np.vstack([H, X])
            self.N_f += self.N_i

        ones = np.ones((1, H.shape[1]))

        if self.bias:
            H = np.vstack([H, ones])
            self.N_f += 1

        if self.reweight:
            linear_weight = 1
            pre_weight = self.col_softmax(np.vstack([H[:self.N_h], linear_weight * ones]))
            H[:self.N_h] = pre_weight[:-1]
            H[self.N_h:] *= pre_weight[-1:]

        return H

    def train(self, X, Y):
        self.N_i = X.shape[0]
        self.N_o = Y.shape[0]

        #         self.W_i = np.random.uniform(-self.sigma,self.sigma, (self.N_h, self.N_i))
        #         self.W_i = X.T[np.random.choice(X.shape[1],self.N_h,replace=False)]
        if self.N_h <= 0:
            self.W_i = np.empty((0, self.N_i))
        else:
            self.W_i, _ = kmeans2(X.T, self.N_h, minit='points')

        H = self._pre_output(X)

        self.W_o = Y.dot(H.T.dot(np.linalg.inv(H.dot(H.T) + self.beta * np.eye(self.N_f))))

    def predict(self, X):
        H = self._pre_output(X)
        return self.W_o.dot(H)

    def predict_multistep(self, X, horizon):
        Y = np.empty((self.N_o * horizon, X.shape[1]))
        Z = np.vstack([X, Y])
        start = X.shape[0]
        Z[start:start + self.N_o, :] = self.predict(X)
        for i in range(1, horizon):
            Z[start + i * self.N_o: start + (i + 1) * self.N_o, :] = self.predict(
                Z[i * self.N_o: start + i * self.N_o, :])
        return Z[start:, :]
