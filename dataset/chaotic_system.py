import numpy as np
import numba
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rescale(x, recover=False):
    x = np.atleast_2d(x)
    row_max = np.max(x, axis=1).reshape((x.shape[0], 1))
    row_min = np.min(x, axis=1).reshape((x.shape[0], 1))
    if recover:
        return (x - row_min) / (row_max - row_min), (row_min, row_max)
    else:
        return 2*(x - row_min) / (row_max - row_min)-1



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
        lorenz=Lorenz,
        chen=Chen,
        chua=Chua,
        switch=Switch,
        three_scroll=three_scroll,
        four_scroll=four_scroll
    )

    if name not in chaotic_systems:
        raise Exception("Invalid system model: {}. Must be one of {}".
                        format(name, list(chaotic_systems.keys())))

    return chaotic_systems[name]()


def Rossler():
    A = 0.2
    B = 0.2
    C = 5.7
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

    return [f1, f2, f3], [x0, y0, z0], 1e-2


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

    return [f1, f2, f3], [x0, y0, z0], 1e-2


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

    return [f1, f2, f3], [x0, y0, z0], 1e-2


def Lorenz():
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

    return [f1, f2, f3], [x0, y0, z0], 1e-2


def Chen():
    a = 40.
    b = 3.
    c = 28.

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

def three_scroll():
    a = 0.977
    b = 10
    c = 4
    d = 0.1
    x0, y0, z0 = 0.01, 0.01, 0.01

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return a * (x - y) - y*z

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return -b*y +x*z

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return -c*z + d*x + x*y

    return [f1, f2, f3], [x0, y0, z0], 1e-2

def four_scroll():
    a = 1.46
    b = 9
    c = 5
    d = 0.06
    x0, y0, z0 = 0.01, 0.01, 0.01

    def f1(X, t):
        x, y, z = X[0], X[1], X[2]
        return a * (x - y) - y*z

    def f2(X, t):
        x, y, z = X[0], X[1], X[2]
        return -b*y +x*z

    def f3(X, t):
        x, y, z = X[0], X[1], X[2]
        return -c*z + d*x + x*y

    return [f1, f2, f3], [x0, y0, z0], 1e-2

if __name__ == '__main__':
    names = [
        'rossler',
        'rabinovich_fabrikant',
        'lorenz',
        'chua',
        # 'three_scroll',
        # 'four_scroll'
    ]

    N = 100000

    for system_name in names:
        print(system_name)

        functions, start_point, step = gen_model(system_name)
        x = trajectory(functions, start_point, N, step)
        x = rescale(x)

        np.savetxt(system_name + '.txt', x.T, fmt='%.8e', delimiter=',')
#         x = np.array([[1,0,0]]).dot(x)
#         x, min_max1d = rescale(x, recover=True)
#         np.savetxt(system_name+'1d.txt', x, fmt='%.8e',delimiter=',')

#         scales = np.vstack([np.hstack(min_max3d),
#                             np.hstack(min_max1d)])
#         np.savetxt(system_name+'_recover.txt', scales, fmt='%.8e',delimiter=',')





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
