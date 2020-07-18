from dataset.chaotic_system import *
import numpy as np
from utils import *
import tqdm

names = [
    'rossler',
    'rabinovich_fabrikant',
    'lorentz',
    'chua',
]


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


def show_3d(x, save=None):
    x = np.atleast_2d(x)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(*x[:3, :], 'green')
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(save)


def col_normalize(x, f):
    if x.shape[0] == 0:
        return x
    fx = f(x)
    fx_sum = fx.sum(axis=0).reshape([1, fx.shape[1]])
    return fx / fx_sum

def recover(x, row_min, row_max):
   return  x * (row_max - row_min) + row_min

def transform(x, row_min, row_max):
    return (x - row_min) / (row_max - row_min)

for system_name in names:
    x3d = np.loadtxt('dataset/'+system_name+'.txt', delimiter=',')
    # x1d = np.loadtxt('dataset/' + system_name + '1d.txt', delimiter=',')
    x1d = np.loadtxt(system_name+'1d_pred.txt', delimiter=',')
    # x1d = np.loadtxt(system_name+'1d_test.txt', delimiter=',')
    scales = np.loadtxt('dataset/'+system_name+'_recover.txt', delimiter=',')
    C = np.array([[0.3, 0.3, 0.4]]).T

    test_start = 5000 + 10  # test_start + horizon
    x3d = recover(x3d, scales[:3, 0], scales[:3,1])
    x_pred = np.zeros((len(x1d), 3))
    x_pred[0] = x3d[test_start]


    observations = transform(x_pred[0], scales[:3, 0], scales[:3, 1]).dot(C)
    observations = transform(observations, scales[-1, 0], scales[-1, 1])
    initial_error = np.abs(observations - x1d[0])
    print('initial_error',initial_error)
    # break

    functions, _, step_size = gen_model(system_name)

    N_particles = 1000
    radius = 1e-2
    h = 5
    particles = np.empty((N_particles, h+1, 3))
    N_decode = 1000
    decode_error = []

    for t in tqdm.trange(1,N_decode):
        center = x_pred[t-1]

        particles[:, 0] = center + [np.random.randn(*center.shape) * radius for _ in range(N_particles)]
        particles[0, 0] = center

        for i in range(N_particles):

            for hi in range(h):
                dX = dxdt(functions, particles[i,hi], np.nan, step_size)
                particles[i,hi+1] = particles[i,hi] + dX

        observations = particles.reshape((-1, 3))
        observations = transform(observations, scales[:3, 0], scales[:3,1]).dot(C)
        observations = transform(observations, scales[-1, 0], scales[-1, 1])
        observations = observations.reshape((N_particles, h+1))

        weights = -np.log(np.sum(np.abs(observations[:,] - x1d[t-1:t+h]),axis=1))
        # print(weights)
        # average = np.sum(particles * weights, axis=0)
        best_choice = particles[np.argmax(weights)][1]
        # print(average, average-x3d[t+test_start], np.mean(np.square(average - x3d[t+test_start])))
        # decode_error.append(np.mean(np.square(average - x3d[t+test_start])))
        x_pred[t] = best_choice
        # break

    plt.figure(figsize=(20, 6))
    dim = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(4, 1, i + 1)
        # plt.plot(x3d.T[i, test_start:test_start+N_decode], color='black', label='ground truth')
        plt.plot(x_pred.T[i, :N_decode], color='red', label='recovered')
        plt.ylabel(dim[i])
        plt.legend(loc='upper right')
    plt.subplot(4, 1, 4)
    plt.plot(decode_error, color='black')
    plt.ylabel('decode error(MSE)')
    plt.xlabel('t')
    plt.savefig('../' + system_name + '_1drecover.pdf')


    expected = transform(x_pred, scales[:3, 0], scales[:3, 1]).dot(C)
    expected = transform(expected, scales[-1, 0], scales[-1, 1])
    recover_error = x1d - np.squeeze(expected.T)
    recover_error = recover_error[:N_decode]
    plt.figure()
    plt.plot(recover_error)
    plt.savefig('../' + system_name + '_1derror.pdf')

    show_3d(x_pred[:N_decode].T, save='../' + system_name + '_3drecover.pdf')

plt.show()


