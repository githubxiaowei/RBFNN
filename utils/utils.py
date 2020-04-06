import numpy as np
from numba import jit
from skfuzzy.cluster import cmeans
from itertools import product

def generate_conf(confs):
    """
    generate parameters from dict
    :param confs:
    confs = {
        'param1':[1,2, ..., 9],
        ...
        'param9':[1,2,..., 9]
    }
    :return:
    """
    for conf in product(*confs.values()):
        yield {k:v for k,v in zip(confs.keys(),conf)}


def rmse(A, B):
    return np.sqrt(np.sum(np.square(A - B)))


def mse(A, B):
    return np.mean(np.square(A - B))


def smape(A, B):
    A_norm1 = np.sum(np.abs(A))
    B_norm1 = np.sum(np.abs(B))
    return 2 * np.sum(np.abs(A - B)) / (A_norm1 + B_norm1)


def mape(A, B):
    A_norm1 = np.abs(A)
    B_norm1 = np.abs(B)
    return np.sum(np.abs(A - B) / B_norm1)


def error_multistep(err_func, A, B, dim=3):
    """
    A, B: np.array with shape (dim * horizon, 1)
    """
    # print(A.shape, B.shape)
    assert (A.shape == B.shape)
    horizon = A.shape[0] // dim
    mse_at_each_horizon = []
    for i in range(horizon):
        mse_at_each_horizon.append(err_func(A[i * dim:(i + 1) * dim], B[i * dim:(i + 1) * dim]))
    return mse_at_each_horizon


def rescale(x):
    row_max = np.max(x, axis=1).reshape((x.shape[0], 1))
    row_min = np.min(x, axis=1).reshape((x.shape[0], 1))
    return (x - row_min) / (row_max - row_min)


def select_samples(X, start, num):
    return X[:, start:start + num]


def fcm(data, c, m=2):
    center, u, u0, d, jm, p, fpc = cmeans(data.T, c, m=m, error=1e-6, maxiter=20)
    u_m = u ** m
    D = np.sqrt(np.sum(np.square(data[:, None] - center), axis=2))
    radius = np.asarray([np.sum(u_m[i] @ D[:, i]) / np.sum(u_m[i]) for i in range(c)])
    return center, radius


@jit(nopython=True)
def dtw_distance(ts_a, ts_b, k, mww=2):
    """Computes dtw distance between two time series

    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)

    Returns:
        dtw distance
    """

    # Create cost matrix via broadcasting with large int
    cost = np.ones((k, k))

    # Initialize the first row and column
    cost[0, 0] = abs(ts_a[0] - ts_b[0])
    for i in range(1, k):
        cost[i, 0] = cost[i - 1, 0] + abs(ts_a[i] - ts_b[0])

    for j in range(1, k):
        cost[0, j] = cost[0, j - 1] + abs(ts_a[0] - ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, k):
        for j in range(max(1, i - mww), min(k, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + abs(ts_a[i] - ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


def pairwise_dtw_distances(X, Y):
    m, n, k = X.shape[0], Y.shape[0], X.shape[1]
    # D = np.empty((m, n))
    # for i in range(m):
    #     for j in range(n):
    #         D[i, j] = dtw_distance(X[i][0::3], Y[j][0::3], k//3) + \
    #                   dtw_distance(X[i][1::3], Y[j][1::3], k//3) + \
    #                   dtw_distance(X[i][2::3], Y[j][2::3], k//3)
    if m == 0:
        return np.empty((0, n))

    D = np.asarray([[
        dtw_distance(X[i][0::3], Y[j][0::3], k // 3) +
        dtw_distance(X[i][1::3], Y[j][1::3], k // 3) +
        dtw_distance(X[i][2::3], Y[j][2::3], k // 3)
        for j in range(n)] for i in range(m)]
    )
    return D
