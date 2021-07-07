#!/usr/bin/env python
# coding: utf-8

# In[428]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import random
import scipy.linalg
import seaborn as sns
import time

from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from scipy import signal
from sklearn.cluster import KMeans



class Timer(object):
    def __enter__(self):
        self.t0 = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[time spent: {time:.2f}s]'.format(time = time.time() - self.t0))

def pairwise_distances(X, Y):
    D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
    D[D < 0] = 0
    return D

path = 'result/esn_attn_20210117'
if not os.path.exists(path):
    os.makedirs(path)


def sample(weights, labels, N_new=None):
    if N_new is None:
        N_new = len(labels)

    N = len(labels)
    p = []
    indice = []
    index = int(random.random() * N)
    beta = 0.0  # 轮盘指针
    mw = np.sum(weights)
    while len(indice) < N_new:
        beta += random.random() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        if not index in indice:
            p.append(labels[index])
            indice.append(index)

    return p, indice

def log(file, content):
    with open(file, 'a') as file:
        file.write(str(content))

for seed in range(100):
    logfile = '{}/{}.txt'.format(path, seed)
    if os.path.exists(logfile):
        os.remove(logfile)
    for system_name in [
        'Rossler',
        'Rabinovich Fabrikant',
        'Lorenz'
    ]:
        print(system_name, seed)
        for ndim in [1]:

            try:
                with Timer():
                    random.seed(seed)
                    np.random.seed(seed)

                    log(logfile, seed)
                    log(logfile, ',')
                    log(logfile, system_name)
                    log(logfile, ',')
                    log(logfile, ndim)
                    log(logfile, ',')

                    print()
                    U0 = np.loadtxt('dataset/'+system_name+'.csv', delimiter=',').T[:ndim]
                    U = U0 + np.random.randn(*U0.shape)*1e-3
                    U = np.atleast_2d(U)


                    num_prepare = 1000
                    train_start = num_prepare
                    num_train = 10000
                    val_start = num_prepare + num_train
                    num_val = 100
                    test_start = num_prepare + num_train + num_val
                    num_test = 5000
                    print('dataset shape:', U.shape)


                    inSize = U.shape[0]
                    outSize = inSize
                    resSize = 1000
                    a = 0.9         # leaking rate
                    K = 0.99         # spectial radius
                    reg = 1e-6       # regularization coefficient
                    input_scaling = 1
                    N_c = 100


                    # generation of random weights
                    Win = (np.random.rand(resSize,1+inSize)-0.5) * input_scaling
                    W = np.random.rand(resSize,resSize)-0.5

                    largest_eigvals, _ = largest_eigsh(W @ W.T, 1, which='LM')
                    rhoW = np.sqrt(largest_eigvals[0])
                    W = W/rhoW*(K-1+a)/a
                    X = np.zeros((resSize,U.shape[1]))
                    x = np.zeros([resSize,1])

                    for t in range(U.shape[1]):
                        u = U[:,t:t+1]
                        x = (1-a) * x + a * np.tanh( Win @ np.vstack((1,u)) + W @ x )
                        X[:,t:t+1] = x


                    # offline train
                    U_train = U[:,train_start : train_start + num_train]
                    X_train = X[:,train_start : train_start + num_train]
                    Y_train = U[:,train_start + 1 : train_start + num_train + 1]

                    Ml, Diag, Mr = np.linalg.svd(X_train, full_matrices=False)

                    N_v = 10

                    MlT, Diag, Mr = Ml.T[:N_v], Diag[:N_v], Mr[:N_v]


                    ######################### ESN #############################
                    log(logfile, 'ESN')
                    log(logfile, ',')

                    H = np.vstack((np.ones((1, U_train.shape[1])), X_train))
                    Wout2 = Y_train @ H.T @ np.linalg.inv(H @ H.T + reg * np.eye(H.shape[0]))
                    # print('pre_output shape:', H.shape)
                    # print('W_out shape:', Wout.shape)

                    horizon = 100
                    mse2 = []
                    for h in range(horizon):
                        if h == 0:
                            U_test = U[:, test_start: test_start + num_test]
                            X_test = X[:, test_start: test_start + num_test]
                        else:
                            U_test = Y_pred
                            X_test = (1 - a) * X_test + a * np.tanh(
                                Win @ np.vstack((np.ones((1, num_test)), U_test)) + W @ X_test)
                        H = np.vstack((np.ones((1, num_test)), X_test))
                        Y_pred = Wout2 @ H
                        Y_true = U[:, test_start + h + 1: test_start + num_test + h + 1]
                        mse2.append(np.sum(np.square(Y_pred - Y_true), axis=0))

                    print('error at   1 step:', np.average(mse2[0]))
                    print('error at 100 step:', np.average(mse2[-1]))
                    log(logfile, ','.join([str(e) for e in np.average(mse2, axis=1)]))
                    log(logfile, ',')



                    ######################### ATTN #######################
                    log(logfile, 'ESN-ATTN')
                    log(logfile, ',')

                    # Cs3 = (MlT@X_train).T[np.random.choice(num_train, N_c,replace=False)]

                    c = Mr[-1] ** 2
                    Cs3, indice = sample(c, (MlT @ X_train).T, N_c)
                    Cs3 = np.array(Cs3)

                    #     Cs3 = np.random.random([N_c, N_v])
                    #     okm = OnlineKMeans(N_v, N_c, flag=True, random_init=False)
                    #     for p in tqdm((MlT@X_train).T):
                    #         okm.fit(p)
                    #     Cs3 = okm.centroid

                    choices = []
                    for beta in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]:
                        S = np.exp(- beta * pairwise_distances(Cs3, (np.diag(Diag)@ Mr).T))
                        # S_trans = S[:,1:] @ S[:,:-1].T @ np.linalg.inv(S[:,:-1] @ S[:,:-1].T + reg * np.eye(N_c))
                        # S = S_trans @ S
                        H = np.vstack((np.ones((1,num_train)), X_train, S))
                        Wout3 = Y_train @ H.T @ np.linalg.inv( H @ H.T + reg*np.eye(H.shape[0]))

                        # validation
                        horizon = 20
                        mse3 = []
                        for h in range(horizon):
                            if h == 0:
                                U_val = U[:,val_start : val_start + num_val]
                                X_val = X[:,val_start : val_start + num_val]
                            else:
                                U_val = Y_pred
                                X_val = (1 - a) * X_val + a * np.tanh( Win @ np.vstack((np.ones((1,num_val)),U_val)) + W @ X_val )
                            S = np.exp( - beta* pairwise_distances(Cs3, (MlT @ X_val).T))
                            H = np.vstack((np.ones((1,num_val)), X_val, S))
                            Y_pred = Wout3 @ H
                            Y_true = U[:,val_start+h+1 : val_start + num_val + h+1]
                            mse3.append(np.average(np.sum(np.square(Y_pred - Y_true), axis=0)))
                        choices.append((mse3[-1], beta, Wout3))
                    train_error, beta3, Wout3 = sorted(choices)[0]

                    # test

                    horizon = 100
                    mse3 = []
                    for h in range(horizon):
                        if h == 0:
                            U_test = U[:,test_start : test_start + num_test]
                            X_test = X[:,test_start : test_start + num_test]
                        else:
                            U_test = Y_pred
                            X_test = (1 - a) * X_test + a * np.tanh( Win @ np.vstack((np.ones((1,num_test)),U_test)) + W @ X_test )
                        S = np.exp( - beta3 * pairwise_distances(Cs3, ( MlT @ X_test).T))
                        H = np.vstack((np.ones((1,num_test)), X_test, S))
                        Y_pred = Wout3 @ H
                        Y_true = U[:,test_start+h+1 : test_start + num_test + h+1]
                        mse3.append(np.sum(np.square(Y_pred - Y_true), axis=0))

                    print('error at   1 step:', np.average(mse3[0]))
                    print('error at 100 step:', np.average(mse3[-1]))
                    log(logfile, ','.join([str(e) for e in np.average(mse3, axis=1)]))
                    log(logfile, ',')




                    log(logfile, '\n')

            except Exception as e:
                print(e)
