import numpy as np
from utils import *
from scipy.cluster.vq import kmeans2
from functools import partial
from networkx import erdos_renyi_graph, adjacency_matrix


class ReservoirEncoder:
    def __init__(self, reservoirConf):
        self.nz = reservoirConf.nz
        self.nu = reservoirConf.nu
        self.alpha = reservoirConf.alpha
        self.target_rho = reservoirConf.target_rho
        input_scale = reservoirConf.input_scale

        # sparse recurrent weights init
        if reservoirConf.connectivity < 1:
            g = erdos_renyi_graph(
                reservoirConf.nz,
                reservoirConf.connectivity,
                seed=42,
                directed=True
            )
            self.A = np.array(adjacency_matrix(g).todense()).astype(np.float)

        # full-connected recurrent weights init
        else:
            self.A = np.random.uniform(-1, +1, size=(self.nz, self.nz))

        rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= self.target_rho / rho
        self.B = np.random.uniform(-input_scale, input_scale, size=(self.nz, self.nu))

    def transform(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nx // self.nu):
            u = x[i * self.nu:(i + 1) * self.nu]
            z = (1 - self.alpha) * z + self.alpha * np.tanh(self.A @ z + self.B @ u)
        # z = rescale(z)
        return z

    def echostate(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nt):
            u = x[-self.nu:, i]
            z[:, i] = (1 - self.alpha) * z[:, i - 1] + self.alpha * np.tanh(self.A @ z[:, i - 1] + self.B @ u)
        # z = rescale(z)
        return z


class RBFNN:

    def __init__(self, **kwargs):
        self.conf = kwargs
        self.skip_con = kwargs.get('skip_con', 1)
        self.bias = kwargs.get('bias', 1)
        self.N_i = None
        self.N_o = None
        self.N_h = kwargs.get('N_h', 1)
        self.N_f = self.N_h
        self.sigma = kwargs.get('sigma', 1)
        self.beta = 1e-6
        self.sigmas = np.ones((self.N_h, 1))
        self.method = kwargs.get('method', 'ridge')
        self.activation = kwargs.get('activatioin', 'rbf')
        self.W_i, self.sigmas = kwargs.get('centers', (None, None))
        self.reservoirConf = kwargs.get('reservoirConf', None)
        self.REncoder = ReservoirEncoder(self.reservoirConf) if self.reservoirConf is not None else None
        if self.REncoder:
            self.use_reseroir_state = kwargs.get('use_reseroir_state', False)
            self.encoder_type = kwargs.get('encoder_type', 1)
            self.encoder_func = self.REncoder.transform if self.encoder_type == 1 else self.REncoder.echostate

    @staticmethod
    def pairwise_distances(X, Y):
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return D

    @staticmethod
    def col_normalize(x, f):
        if x.shape[0] == 0:
            return x
        fx = f(x)
        fx_sum = fx.sum(axis=0).reshape([1, fx.shape[1]])
        return fx / fx_sum

    def _pre_output(self, X, Z=None):
        # 隐层输出
        if self.N_h > 0:
            if self.activation == 'rbf' or self.activation == 'rbf-random':
                H = np.exp(
                    -self.pairwise_distances(self.W_i, Z.T) /
                    (2 * self.sigma ** 2 * self.sigmas.reshape((-1, 1)))
                )
            elif self.activation == 'sigmoid':
                H = np.tanh(self.W_i.dot(np.vstack([X, np.ones([1, X.shape[1]])])))

        else:
            H = np.empty((0, X.shape[1]))

        if self.REncoder and self.use_reseroir_state:
            H = np.vstack([H, Z])
            self.N_f += Z.shape[0]

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

        Z = self.encoder_func(X) if self.REncoder else X

        if self.N_h <= 0:
            self.W_i = np.empty((0, self.N_i))

        else:
            # self.W_i, _ = kmeans2(X.T, self.N_h, minit='points')
            if self.activation == 'rbf':
                if self.W_i is None:
                    self.W_i, self.sigmas = fcm(Z.T, self.N_h)
                    # print('generate centers by FCM')

            elif self.activation == 'rbf-random' or self.activation == 'sigmoid':
                self.W_i = np.random.uniform(-1, 1, (self.N_h, self.N_i))
                # self.W_i = X.T[np.random.choice(X.shape[1],self.N_h,replace=False)]

        H = self._pre_output(X, Z)

        if self.method == 'ridge':
            self.W_o = ridge_regressor(H, Y, self.beta)
        elif self.method == 'gradient':
            linear_slice = [i for i in range(self.N_h, self.N_h + self.N_i)] if self.skip_con else []
            self.W_o = gradient_descent(H, Y, [])

    def predict(self, X):
        Z = self.encoder_func(X) if self.REncoder else X
        H = self._pre_output(X, Z)
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
            # print(conf)
            x_train, y_train = train
            x_test, y_test = test

            model = RBFNN(**conf)

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

    @staticmethod
    def parameter_selecting(confs, trainset, valset):
        """
        select best model parameters
        :param trainset:  (x_train, y_train)
        :param valset: (x_val, Y_val)
        :return: configurations with smallest err
        """
        result = RBFNN.grid_search(trainset, valset, confs)
        err = []
        for i in range(len(result)):
            line, conf = result[i][0], result[i][1]
            err.append(line[-1])
        best_one = np.argmin(err)
        return result[best_one][1]

    @staticmethod
    def prepare_train(series, n_history=5):
        """
        :param series:
        :param n_history: num of history points used for inputs
        :return:
        """
        num_train = series.shape[1] - n_history
        horizon = 1
        x_train = np.vstack([select_samples(series, i, num_train) for i in range(n_history)])
        y_train = np.vstack([select_samples(series, n_history + i, num_train) for i in range(horizon)])
        assert (x_train.shape[1] == y_train.shape[1])
        return [x_train, y_train]

    @staticmethod
    def prepare_val(series, n_history=5, horizon=30):
        num_val = series.shape[1] - horizon - n_history + 1
        x_val = np.vstack([select_samples(series, i, num_val) for i in range(n_history)])
        y_val = np.vstack([select_samples(series, n_history + i, num_val) for i in range(horizon)])
        assert (x_val.shape[1] == y_val.shape[1])
        return [x_val, y_val]

    @staticmethod
    def split_train_validation(k_fold, series, **kwargs):
        horizon = kwargs.get('horizon', 30)
        n_history = kwargs.get('n_history', 5)

        n_samples = series.shape[1]
        n_val = n_samples // k_fold

        trainset = RBFNN.prepare_train(series[:, :-n_val], n_history=n_history)
        valset = RBFNN.prepare_val(series[:, -n_val:], n_history=n_history, horizon=horizon)

        return trainset, valset
