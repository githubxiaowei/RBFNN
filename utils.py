import numpy as np
from numba import jit

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
        cost[i, 0] = cost[i - 1, 0] + abs(ts_a[i]-ts_b[0])

    for j in range(1, k):
        cost[0, j] = cost[0, j - 1] + abs(ts_a[0]-ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, k):
        for j in range(max(1, i - mww), min(k, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + abs(ts_a[i]- ts_b[j])

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
        return np.empty((0,n))

    D = np.asarray([[
                      dtw_distance(X[i][0::3], Y[j][0::3], k//3) +
                      dtw_distance(X[i][1::3], Y[j][1::3], k//3) +
                      dtw_distance(X[i][2::3], Y[j][2::3], k//3)
                     for j in range(n)] for i in range(m)]
                   )
    return D