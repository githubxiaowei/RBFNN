import numpy as np
from numba import jit
from skfuzzy.cluster import cmeans
from itertools import product

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'
}


def select_samples(X, start, num):
    return np.atleast_2d(X)[:, start:start + num]


def rescale(x):
    x = np.atleast_2d(x)
    row_max = np.max(x, axis=1).reshape((x.shape[0], 1))
    row_min = np.min(x, axis=1).reshape((x.shape[0], 1))
    return (x - row_min) / (row_max - row_min)


def  extract_num(conf):
    numstr = conf.split(' ')[-1]
    parts = numstr.split('/')
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return None
    num = float(parts[0])/float(parts[1])
    return num

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


def fcm(data, c, m=2):
    center, u, u0, d, jm, p, fpc = cmeans(data.T, c, m=m, error=1e-6, maxiter=20)
    # u_m = u ** m
    u_m = u
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
