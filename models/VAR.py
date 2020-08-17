from utils import *


class VAR:

    def __init__(self, **kwargs):
        self.conf = kwargs
        self.bias = True
        self.N_i = None
        self.N_o = None
        self.N_f = 0
        self.beta = 1e-6
        self.method = 'ridge'
        self.skip_con = 1

    def _pre_output(self, X):
        """
        :param X: (N_i, N_samples)
        :return: H: (..., N_samples)
        """

        H = np.empty((0, X.shape[1]))
        self.N_f = 0

        H = np.vstack([H, X])
        self.N_f += self.N_i

        if self.bias:
            ones = np.ones((1, H.shape[1]))
            H = np.vstack([H, ones])
            self.N_f += 1

        return H

    def train(self, X, Y, **kwargs):

        self.N_i = X.shape[0]
        self.N_o = Y.shape[0]

        H = self._pre_output(X)

        self.W_o = ridge_regressor(H, Y, self.beta)

    def predict(self, X, **kwargs):
        H = self._pre_output(X)
        return self.W_o.dot(H)

    def predict_multistep(self, X, horizon, **kwargs):
        Y = np.empty((self.N_o * horizon, X.shape[1]))
        Z = np.vstack([X, Y])
        start = X.shape[0]
        Z[start:start + self.N_o, :] = self.predict(X)
        for i in range(1, horizon):
            Z[start + i * self.N_o: start + (i + 1) * self.N_o, :] = self.predict(
                Z[i * self.N_o: start + i * self.N_o, :])
        return Z[start:, :]


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
            model.train(x_train, y_train)
            Predictions = model.predict(x_test)
            MSE[j] = mse(Predictions, y_test)
        best_model = model_confs[np.argmin(MSE)]
        return best_model


    n_dim = 1
    horizon = 1
    N = 10000
    n_history = 10  # 使用 n 个历史点作为输入
    num_prepare = 0
    num_train = 2000
    num_test = 2000
    train_start = 0
    test_start = 5000

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
            [('VAR', VAR, dict())]
        )

        for j, confs in enumerate(model_confs):
            conf = select_model(confs)
            model_name, model_class, kwargs = conf
            model = model_class(**kwargs)
            model.train(x_train, y_train)
            Predictions = model.predict(x_test)
            MSE = mse(Predictions, y_test)
            print(MSE)
