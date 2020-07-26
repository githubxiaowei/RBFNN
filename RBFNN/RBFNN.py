import numpy as np
from utils import *
from scipy.cluster.vq import kmeans2
from functools import partial
from networkx import erdos_renyi_graph, adjacency_matrix
from enum import Enum

class ReservoirEncoder:
    def __init__(self, reservoirConf):
        self.nz = reservoirConf.nz
        self.nu = reservoirConf.nu
        self.alpha = reservoirConf.alpha
        self.target_rho = reservoirConf.target_rho
        input_scale = reservoirConf.input_scale
        self.state = None

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

    def state_transition(self, z, u):
        z = (1 - self.alpha) * z + self.alpha * np.tanh(self.A @ z + self.B @ u)
        return z

    def transform(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nx // self.nu):
            u = x[i * self.nu:(i + 1) * self.nu]
            z = self.state_transition(z, u)
        return z

    def echostate(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nt):
            u = x[-self.nu:, i]
            z[:, i] = self.state_transition(z[:, i-1], u)
        return z

class ModelType(Enum):
    VAR = 1
    ESN = 2
    RBFN = 3
    RBFLN = 4
    RBFLN_RE = 5
    ESN_ATTN = 6


class RBFNN:

    def __init__(self, model_type, **kwargs):
        self.conf = kwargs
        self.model_type = model_type
        self.REncoder = None
        self.sigmas = None
        self.bias = True
        self.W_i = None
        self.N_i = None
        self.N_o = None
        self.N_f = 0
        self.beta = 1e-6
        self.method = 'ridge'
        self.prepare_reservoir = False

        if self.model_type == ModelType.RBFLN:
            self.skip_con = 1
            self.activation = 'rbf'
            self.N_h = kwargs.get('N_h')
            self.sigma = kwargs.get('sigma', 1)

        elif self.model_type == ModelType.VAR:
            self.skip_con = 1
            self.N_h = 0

        elif self.model_type == ModelType.RBFN:
            self.skip_con = 0
            self.activation = 'rbf'
            self.N_h = kwargs.get('N_h')
            self.sigma = kwargs.get('sigma', 1)

        elif self.model_type == ModelType.ESN:
            self.skip_con = 1
            self.N_h = 0
            self.reservoirConf = kwargs.get('reservoirConf')
            self.REncoder = ReservoirEncoder(self.reservoirConf)
            self.encoder_func = self.REncoder.echostate
            self.prepare_reservoir = True

        elif self.model_type == ModelType.RBFLN_RE:
            self.skip_con = 1
            self.activation = kwargs.get('activation', 'rbf')
            self.N_h = kwargs.get('N_h')
            self.sigma = kwargs.get('sigma', 1)
            self.reservoirConf = kwargs.get('reservoirConf')
            self.REncoder = ReservoirEncoder(self.reservoirConf)
            # self.encoder_func = self.REncoder.echostate
            self.encoder_func = self.REncoder.transform if kwargs.get('encoder', 'transform') == 'transform' else self.REncoder.echostate

        elif self.model_type == ModelType.ESN_ATTN:
            self.skip_con = 1
            self.activation = 'rbf'
            self.N_h = kwargs.get('N_h')
            self.sigma = kwargs.get('sigma', 1)
            self.reservoirConf = kwargs.get('reservoirConf')
            self.REncoder = ReservoirEncoder(self.reservoirConf)
            self.encoder_func = self.REncoder.echostate
            self.prepare_reservoir = True

    @staticmethod
    def pairwise_distances(X, Y):
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return D

    @staticmethod
    def col_normalize(x, f=None):
        if x.shape[0] == 0:
            return x

        fx = f(x) if f else x
        fx_sum = fx.sum(axis=0).reshape([1, fx.shape[1]])
        return fx / fx_sum

    def _pre_output(self, X, Z=None):
        """

        :param X: (N_i, N_samples)
        :param Z: (nz, N_samples)
        :param W_i: ((N_h, nz)
        :return: H: (..., N_samples)
        """

        if self.N_h > 0:
            H = np.exp(
                -self.pairwise_distances(self.W_i, Z.T) /
                (2 * self.sigma ** 2 * self.sigmas.reshape((-1, 1)))
            )  # (N_h, N_samples)
        else:
            H = np.empty((0, X.shape[1]))
        self.N_f = max(0, self.N_h)

        if self.REncoder:
            if self.model_type == ModelType.ESN:
                H = Z
                self.N_f = Z.shape[0]
            elif self.model_type == ModelType.ESN_ATTN:
                assert(self.N_h > 0)
                H = self.col_normalize(H)
                H = self.W_i.T @ H
                H = np.vstack([H, Z])
                self.N_f = 2*Z.shape[0]

        if self.skip_con:
            H = np.vstack([H, X])
            self.N_f += self.N_i

        if self.bias:
            ones = np.ones((1, H.shape[1]))
            H = np.vstack([H, ones])
            self.N_f += 1

        return H

    def train(self, X, Y, num_prepare):

        self.N_i = X.shape[0]
        self.N_o = Y.shape[0]

        Z = self.encoder_func(X) if self.REncoder else X
        X, Z = X[:, num_prepare:], Z[:, num_prepare:]

        if self.N_h <= 0:
            self.W_i = np.empty((0, self.N_i))

        else:
            # self.W_i, _ = kmeans2(X.T, self.N_h, minit='points')
            if self.activation == 'rbf':
                if self.W_i is None:
                    self.W_i, self.sigmas = fcm(Z.T, self.N_h)

            elif self.activation == 'rbf-random':
                self.W_i = np.random.uniform(-1, 1, (self.N_h, self.N_i))
                # self.W_i = X.T[np.random.choice(X.shape[1],self.N_h,replace=False)]

        H = self._pre_output(X, Z)

        self.W_o = ridge_regressor(H, Y, self.beta)


    def predict(self, X, num_prepare=0):

        Z = self.encoder_func(X) if self.REncoder else X
        X, Z = X[:, num_prepare:], Z[:, num_prepare:]

        H = self._pre_output(X, Z)
        return self.W_o.dot(H)

    def predict_multistep(self, X, horizon):
        Y = np.empty((self.N_o * horizon, X.shape[1]))
        Z = np.vstack([X, Y])
        start = X.shape[0]
        Z[start:start + self.N_o, :] = self.predict(X)
        for i in range(1, horizon):
            Z[start + i * self.N_o: start + (i + 1) * self.N_o, :] = self.predict(Z[i * self.N_o: start + i * self.N_o, :])
        return Z[start:, :]

    def predict_multistep_esn(self, X, horizon):
        Y = np.empty((self.N_o * horizon, 1))
        Z = self.encoder_func(X)
        z = Z[:, -1:]
        x = X[:, -1:]
        for i in range(horizon):
            H = self._pre_output(x, z)
            y = self.W_o.dot(H)
            Y[i * self.N_o: (i + 1) * self.N_o] = y
            x = np.vstack([x, y])[-self.N_i:]
            z = self.REncoder.state_transition(z, y)
        print(X[:,-2:])
        print(Y[:2])
        return Y


