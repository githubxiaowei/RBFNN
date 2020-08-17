from utils import *
from models.Reservoir import ReservoirEncoder

class RBFN_RE:

    def __init__(self, **kwargs):
        self.conf = kwargs

        self.sigmas = None
        self.bias = True
        self.W_i = None
        self.N_i = None
        self.N_o = None
        self.N_f = 0
        self.beta = 1e-6
        self.method = 'ridge'
        self.skip_con = kwargs.get('skip_con', 0)
        self.activation = kwargs.get('activation', 'rbf')
        self.N_h = kwargs.get('N_h')
        self.sigma = kwargs.get('sigma', 1)
        self.reservoirConf = kwargs.get('reservoirConf')
        self.REncoder = ReservoirEncoder(self.reservoirConf)
        self.encoder_func = self.REncoder.transform \
            if kwargs.get('encoder', 'transform') == 'transform' \
            else self.REncoder.echostate


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

        if self.skip_con:
            H = np.vstack([H, X[-self.skip_con:]])
            self.N_f += len(X[-self.skip_con:])

        if self.bias:
            ones = np.ones((1, H.shape[1]))
            H = np.vstack([H, ones])
            self.N_f += 1

        return H

    def train(self, X, Y, **kwargs):

        num_prepare = kwargs.get('num_prepare', 0)

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


    def predict(self, X, **kwargs):
        num_prepare = kwargs.get('num_prepare', 0)
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
        return Y



class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


if __name__ == '__main__':

    names = [
        'rossler',
        'rabinovich_fabrikant',
        'lorentz',
        'chua',
    ]
    names = [n + '1d' for n in names]


    def gen_model(conf):
        _, model_class, kwargs = conf
        return model_class(**kwargs)


    def select_model(model_confs):
        MSE = [0.0] * len(model_confs)
        for j, conf in enumerate(model_confs):
            model = gen_model(conf)
            model.train(x_train, y_train, num_prepare=num_prepare)
            Predictions = model.predict(x_test, num_prepare=num_prepare)
            MSE[j] = mse(Predictions, y_test)
        best_model = model_confs[np.argmin(MSE)]
        return best_model


    n_dim = 1
    horizon = 1
    N = 10000
    n_history = 10  # 使用 n 个历史点作为输入
    num_prepare = 20
    num_train = 2000
    num_test = 2000
    train_start = 0
    test_start = 5000
    N_h = 200
    np.random.seed()
    nz = 100
    reservoirConf = Dict(
        alpha=0.9,
        connectivity=1,
        nz=nz,
        nu=n_dim,
        target_rho=0.99,
        input_scale=1,
        #     activation = lambda x: 1/(1 + np.exp(-x))  # sigmoid
        #     activation = lambda x: np.maximum(0,x)
        #     activation = lambda x: x/(1 + np.exp(-x))
        activation=np.tanh
        #     activation = lambda x: np.sin(x)
    )

    for system_name in names:

        print(system_name)
        '''
        数据集
        '''
        x = np.loadtxt('../dataset/' + system_name + '.txt', delimiter=',').T
        x += np.random.randn(*x.shape) * 0.001

        x_train = np.vstack(
            [select_samples(x, train_start + i, num_train + num_prepare)
             for i in range(n_history)])
        x_test = np.vstack(
            [select_samples(x, test_start + i, num_test + num_prepare)
             for i in range(n_history)])
        y_test = select_samples(x, test_start + num_prepare + n_history + horizon - 1, num_test)
        y_train = select_samples(x, train_start + num_prepare + n_history + horizon - 1, num_train)

        model_confs = []

        model_confs.append(
            [('RBFN-RE-transform',
              RBFN_RE,
              dict(N_h=N_h, sigma=sigma, reservoirConf=reservoirConf, encoder='transform', skip_con=1))
             for sigma in [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]]
        )

        model_confs.append(
            [('RBFN-RE-echostate',
              RBFN_RE,
              dict(N_h=N_h, sigma=sigma, reservoirConf=reservoirConf, encoder='echostate', skip_con=1))
             for sigma in [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]]
        )

        for j, confs in enumerate(model_confs):
            conf = select_model(confs)
            model_name, model_class, kwargs = conf
            model = model_class(**kwargs)
            model.train(x_train, y_train, num_prepare=num_prepare)
            Predictions = model.predict(x_test, num_prepare=num_prepare)
            MSE = mse(Predictions, y_test)
            print(MSE)

