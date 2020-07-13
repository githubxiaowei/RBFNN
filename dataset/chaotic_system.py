import numpy as np
import numba
from math import *
from utils import rescale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dxdt(F, X, t, h=1e-2):
    assert(len(F )==len(X))
    X = np.array(X)
    K1 = np.array([f(X, t) for f in F])
    dX = h* K1 / 2
    K2 = np.array([f(X + dX, t + h / 2) for f in F])
    dX = h * K2 / 2
    K3 = np.array([f(X + dX, t + h / 2) for f in F])
    dX = h * K3
    K4 = np.array([f(X + dX, t + h) for f in F])

    dX = (K1 + 2 * K2 + 2 * K3 + K4) * h / 6

    return dX


def trajectory(F, initial_point, num_points=1e4, h=1e-2):
    assert (len(F) == len(initial_point))

    n = int(num_points)
    dim = len(initial_point)
    X = np.zeros([n, dim])

    X[0, :] = initial_point
    for k in range(1, n):
        dX = dxdt(F, X[k - 1, :], h * (k - 1), h)
        X[k, :] = X[k - 1, :] + dX

    return X.T


def gen_model(name: str):
    chaotic_systems = dict(
        rossler=Rossler,
        rabinovich_fabrikant=RabinovichFabrikant,
        lorentz=Lorentz,
        chen=Chen,
        chua=Chua,
        switch=Switch
    )

    if name not in chaotic_systems:
        raise Exception("Invalid system model: {}. Must be one of {}".
                        format(name, list(chaotic_systems.keys())))

    return chaotic_systems[name]()


def Rossler():
    A = 0.4
    B = 0.5
    C = 4.5
    x0, y0, z0 = -2.0, 2.0, 0.2

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return -y - z

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return x + A * y

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return B + z * (x - C)

    return [f1, f2, f3], [x0, y0, z0], 1e-1


def Switch():
    '''
    5 vortex
    '''

    def g(x):
        return np.sum([np.sign(x + 2 * k - 1) + np.sign(x - 2 * k + 1) for k in range(1, 3)]) / 2

    a = -4
    b = 0.2
    c = -5
    x0, y0, z0 = 0.1, 0.1, 0.1

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return a * y

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return x - g(x + z) + b * y

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return c * (z - g(x + z))

    return [f1, f2, f3], [x0, y0, z0], 1e-1


def RabinovichFabrikant():
    ParamA = 1.1
    ParamB = 0.87
    x0, y0, z0 = -1, 0, 0.5

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return y * (z - 1 + x * x) + ParamB * x

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return x * (3 * z + 1 - x * x) + ParamB * y

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return -2 * z * (ParamA + x * y)

    return [f1, f2, f3], [x0, y0, z0], 1e-1


def Chua():
    '''
    3 vortex chua
    '''

    def g(x):
        return 0.6 * x - 1.1 * x * fabs(x) + 0.45 * x ** 3

    a = 12.8
    b = 19.1
    x0, y0, z0 = 1.7, 0.0, -1.9

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return a * (y - g(x))

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return x - y + z

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return -b * y

    return [f1, f2, f3], [x0, y0, z0], 1e-1


def Lorentz():
    '''
    2 vortex lorentz
    '''
    C = 8 / 3
    B = 28
    A = 10

    x0, y0, z0 = -2.0, -3.7, 20.1

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return A * (y - x)

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return B * x - y - x * z

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return x * y - C * z

    return [f1, f2, f3], [x0, y0, z0], 5e-2


def Chen():
    a = 40.
    b = 3.
    c = 28.
    x0 = -0.1
    y0 = 0.5
    z0 = -0.6

    x0, y0, z0 = 8.4, 7.7, 18.5

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return a * (y - x)

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return (c - a) * x - x * z + c * y

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return x * y - b * z

    return [f1, f2, f3], [x0, y0, z0], 2e-2

if __name__ == '__main__':
    names = [
        'rossler',
        'rabinovich_fabrikant',
        'lorentz',
        'chen',
        'chua',
        'switch'
    ]

    N = 10000

    for system_name in names:
        print(system_name)

        functions, start_point, step = gen_model(system_name)
        x = trajectory(functions, start_point, N, step)
        x = rescale(x).T

        print(x.shape)
        x = x.dot(np.array([0.3,0.3,0.4]))

        x = rescale(x)
        np.savetxt(system_name+'1d.txt', x, fmt='%.8e',delimiter=',')



    # combine
    # X = []
    # for system_name in ['rossler', 'lorentz']:
    #     print(system_name)
    #
    #     functions, start_point, step = gen_model(system_name)
    #     x = trajectory(functions, start_point, N, step)
    #     x = rescale(x).T  # (10000,3)
    #     X.append(x)
    #
    # X = np.hstack(X)
    #
    # W = np.hstack([np.eye(3)] * 2).T/2
    # print(W)
    # X = X.dot(W)



    def show(x):
        x = np.atleast_2d(x)
        print(x.shape)
        plt.figure(figsize=(20, 6))
        dim = ['x', 'y', 'z']
        for i in range(x.shape[0]):
            plt.subplot(3, 1, i + 1)
            plt.plot(x[i, :].T, color='green', label='train set')
            plt.ylabel(dim[i])
            plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.show()
    # show(x.T)

    def show_3d(x):
        x = np.atleast_2d(x)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(*x[:3, :], 'green', label='train set')
        # plt.plot(*model.W_i[:,:3].T, 'ko',label='hidden layer')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(system_name)
        # plt.savefig('../figures/' + system_name + '_split.pdf')
        plt.show()

    # show_3d(x.T)