import numpy as np
from utils import *
from scipy.cluster.vq import kmeans2


# import fcm, generate_conf


class RBFNN:
    def __init__(self, N_h,
                 skip_con=True,
                 bias=True,
                 reweight=None,
                 sigma=10,
                 method='ridge'):

        self.skip_con = skip_con
        self.bias = bias
        self.reweight = reweight
        self.N_i = None
        self.N_o = None
        self.N_h = N_h
        self.N_f = N_h
        self.sigma = sigma
        self.beta = 1e-6
        self.sigmas = self.sigma * np.ones((self.N_h, 1))
        self.method = method
        # self.linear_weight = lw if lw else 1

    @staticmethod
    def pairwise_distances(X, Y):
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return D

    @staticmethod
    def col_softmax(x):
        if x.shape[0] == 0:
            return x
        x_col_max = x.max(axis=0)
        x_col_max = x_col_max.reshape([1, x.shape[1]])
        x = x - x_col_max
        x_exp = np.exp(x)
        x_exp_col_sum = x_exp.sum(axis=0).reshape([1, x.shape[1]])
        softmax = x_exp / x_exp_col_sum
        return softmax

    @staticmethod
    def col_normalize(x):
        if x.shape[0] == 0:
            return x
        x_sum = x.sum(axis=0).reshape([1, x.shape[1]])
        softmax = x / x_sum
        return softmax

    def _pre_output(self, X):

        # 隐层输出
        if self.N_h > 0:
            H = np.exp(
                -self.pairwise_distances(self.W_i, X.T) /
                self.sigmas.reshape((-1, 1))
            )
        else:
            H = np.empty((0, X.shape[1]))

        if self.reweight is not None:
            func = dict(
                average=self.col_normalize,
                softmax=self.col_softmax
            )[self.reweight]

            # pre_weight = func(np.vstack([H[:self.N_h], self.linear_weight * ones]))
            pre_weight = func(H[:self.N_h])
            H[:self.N_h] = pre_weight
            # H[self.N_h:] *= pre_weight[-1:]

        if self.skip_con:
            H = np.vstack([H, X])
            self.N_f += self.N_i

        ones = np.ones((1, H.shape[1]))

        if self.bias:
            H = np.vstack([H, ones])
            self.N_f += 1

        return H

    def train(self, X, Y):
        self.N_i = X.shape[0]
        self.N_o = Y.shape[0]

        if self.N_h <= 0:
            self.W_i = np.empty((0, self.N_i))
        else:
            # self.W_i, _ = kmeans2(X.T, self.N_h, minit='points')
            # self.W_i = np.random.uniform(-self.sigma,self.sigma, (self.N_h, self.N_i))
            # self.W_i = X.T[np.random.choice(X.shape[1],self.N_h,replace=False)]
            self.W_i, self.sigmas = fcm(X.T, self.N_h)

        H = self._pre_output(X)

        if self.method == 'ridge':
            self.W_o = ridge_regressor(H, Y, self.beta)
        elif self.method == 'gradient':
            linear_slice = [i for i in range(self.N_h, self.N_h + self.N_i)] if self.skip_con else []
            self.W_o = gradient_descent(H, Y, [])

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

    @staticmethod
    def grid_search(train, test, confs):
        res = []
        for conf in generate_conf(confs):

            x_train, y_train = train
            x_test, y_test = test

            model = RBFNN(conf['n_neuron'],
                          skip_con=conf['skip_con'],
                          reweight=conf['reweight'],
                          sigma=conf['sigma'],
                          method=conf['method'],
                          )

            model.train(x_train, y_train)

            horizon = y_test.shape[0] // model.N_o
            y_mutistep = model.predict_multistep(x_test, horizon)

            mean_list = []
            for i in range(len(x_test)):
                A = y_mutistep[:, i:i + 1]
                B = y_test[:, i:i + 1]
                err_list = error_multistep(mse, A, B, dim=model.N_o)
                mean_list.append(err_list)

            mean_list = np.average(mean_list, axis=0)
            res.append((mean_list, conf))
        return res
