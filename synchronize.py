#!/usr/bin/env python
# coding: utf-8

# In[428]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        print('[time spent: {time:.2f}s]'.format(time=time.time() - self.t0))

def pairwise_distances(X, Y):
    D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
    D[D < 0] = 0
    return D


def svht(X, sigma=None, sv=None):
    """Return the optimal singular value hard threshold (SVHT) value.
    `X` is any m-by-n matrix. `sigma` is the standard deviation of the
    noise, if known. Optionally supply the vector of singular values `sv`
    for the matrix (only necessary when `sigma` is unknown). If `sigma`
    is unknown and `sv` is not supplied, then the method automatically
    computes the singular values."""

    def omega_approx(beta):
        """Return an approximate omega value for given beta. Equation (5) from Gavish 2014."""
        return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43

    def lambda_star(beta):
        """Return lambda star for given beta. Equation (11) from Gavish 2014."""
        return np.sqrt(2 * (beta + 1) + (8 * beta) /
                       (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))

    try:
        m, n = sorted(X.shape)  # ensures m <= n
    except:
        raise ValueError('invalid input matrix')
    beta = m / n  # ratio between 0 and 1
    if sigma is None:  # sigma unknown
        if sv is None:
            sv = svdvals(X)
        sv = np.squeeze(sv)
        if sv.ndim != 1:
            raise ValueError('vector of singular values must be 1-dimensional')
        return np.median(sv) * omega_approx(beta)
    else:  # sigma known
        return lambda_star(beta) * np.sqrt(n) * sigma


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


class OnlineKMeans:
    """ Online K Means Algorithm """

    def __init__(self,
                 num_features: int,
                 num_clusters: int,
                 lr=None,
                 flag=True,
                 random_init=False
                 ):
        """
        :param num_features: The dimension of the data
        :param num_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param lr: The learning rate of the online k-means (c', t0). If None, then we will use the simplest update
        rule (c'=1, t0=0) as described in the lecture.
        """
        if num_features < 1:
            raise ValueError(f"num_features must be greater or equal to 1!\nGet {num_features}")
        if num_clusters < 1:
            raise ValueError(f"num_clusters must be greater or equal to 1!\nGet {num_clusters}")

        self.num_features = num_features
        self.num_clusters = num_clusters

        self.num_centroids = 0
        self.centroid = np.random.random((num_clusters, num_features)) if random_init else np.zeros(
            (num_clusters, num_features))
        self.cluster_counter = np.zeros(num_clusters)  # Count how many points have been assigned into this cluster
        self.centroid_D = np.zeros((num_clusters, num_clusters))
        self.centroid_dist2neighbor = np.zeros((num_clusters, 1))

        self.num_samples = 0
        self.lr = lr
        self.flag = flag
        self.random_init = random_init

    def fit(self, X):
        """
        Receive a sample (or mini batch of samples) online, and update the centroids of the clusters
        :param X: (num_features,) or (num_samples, num_features)
        :return:
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        for i in range(num_samples):
            self.num_samples += 1
            # Did not find enough samples, directly set it to mean
            if not self.random_init and self.num_centroids < self.num_clusters:
                self.centroid[self.num_centroids] = X[i]
                self.cluster_counter[self.num_centroids] += 1
                self.num_centroids += 1
            else:
                # Determine the closest centroid for this sample
                sample = X[i]
                dist = np.linalg.norm(self.centroid - sample, axis=1)

                if self.flag:
                    self.centroid_D = pairwise_distances(self.centroid, self.centroid) + 100 * np.eye(
                        self.num_clusters)
                    self.centroid_dist2neighbor = np.min(self.centroid_D, axis=1)
                    if np.min(dist) > np.average(self.centroid_dist2neighbor) + np.std(self.centroid_dist2neighbor):
                        to_delete = np.argmin(self.centroid_dist2neighbor)
                        self.centroid[to_delete] = sample
                        self.cluster_counter[to_delete] = 0
                else:
                    centroid_idx = np.argmin(dist)

                    if self.lr is None:
                        self.centroid[centroid_idx] = (self.cluster_counter[centroid_idx] * self.centroid[
                            centroid_idx] +
                                                       sample) / (self.cluster_counter[centroid_idx] + 1)
                        self.cluster_counter[centroid_idx] += 1
                    else:
                        #                     c_prime, t0 = self.lr
                        #                     rate = c_prime / (t0 + self.num_samples)
                        rate = self.lr
                        self.centroid[centroid_idx] = (1 - rate) * self.centroid[centroid_idx] + rate * sample
                        self.cluster_counter[centroid_idx] += 1

    def predict(self, X):
        """
        Predict the cluster labels for each sample in X
        :param X: (num_features,) or (num_samples, num_features)
        :return: Returned index starts from zero
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        num_samples, num_features = X.shape

        clusters = np.zeros(num_samples)
        for i in range(num_samples):
            sample = X[i]
            dist = np.linalg.norm(self.centroid - sample, axis=1)
            clusters[i] = np.argmin(dist)
        return clusters

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: (num_features,) or (num_samples, num_features)
        :return:
        """
        # Because the centroid may change in the online setting, we cannot determine the cluster of each label until
        # we finish fitting.
        self.fit(X)
        return self.predict(X)


def clustering(data, N_c):
    estimator = KMeans(init='random', n_clusters=N_c, n_init=3)
    estimator.fit(data)
    return estimator.cluster_centers_


for seed in range(1, 100):

    for system_name in [
        'Rossler',
        'Rabinovich Fabrikant',
        'Lorenz'
    ]:
        print(system_name, seed)
        random.seed(seed)
        np.random.seed(seed)

        U0 = np.loadtxt('dataset/' + system_name + '.csv', delimiter=',').T[:1]
        U0 = np.atleast_2d(U0)

        U = np.vstack([U0[:,i:i+18000] for i in range(9)])

        Ml1, Diag1, Mr1 = np.linalg.svd(U, full_matrices=False)
        print(Diag1)
        U = Ml1.T @ U
        U += np.diag(np.max(U,axis=1)-np.min(U,axis=1)) @ np.random.random(U.shape)*1e-2


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
        a = 0.9  # leaking rate
        K = 0.99  # spectial radius
        reg = 1e-6  # regularization coefficient
        input_scaling = 1
        N_c = 200

        # generation of random weights
        input_scaling = np.max(U,axis=1)-np.min(U,axis=1)
        Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * np.hstack((1, input_scaling)).T
        W = np.random.rand(resSize, resSize) - 0.5
        # g = nx.erdos_renyi_graph(resSize, D / resSize, seed, True)
        # W = nx.adjacency_matrix(g).todense()

        largest_eigvals, _ = largest_eigsh(W @ W.T, 1, which='LM')
        rhoW = np.sqrt(largest_eigvals[0])
        W = W / rhoW * (K - 1 + a) / a
        X = np.zeros((resSize, U.shape[1]))
        x = np.zeros([resSize, 1])

        for t in range(U.shape[1]):
            u = U[:, t:t + 1]
            x = (1 - a) * x + a * np.tanh(Win @ np.vstack((1, u)) + W @ x)
            X[:, t:t + 1] = x

        # offline train
        U_train = U[:, train_start: train_start + num_train]
        X_train = X[:, train_start: train_start + num_train]
        Y_train = U[:, train_start + 1: train_start + num_train + 1]

        H = np.vstack((np.ones((1, num_train)), X_train))
        Wout = Y_train @ H.T @ np.linalg.inv(H @ H.T + reg * np.eye(H.shape[0]))
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
            Y_pred = Wout @ H
            # Y_pred1 = Ml1[-1] @ Y_pred
            # Y_true = U0[:, test_start + h + 9: test_start + num_test + h + 9]
            Y_true = U[:, test_start + h + 1: test_start + num_test + h + 1]
            mse2.append(np.sum(np.square(Y_pred - Y_true), axis=0))


        print('error at   1 step:', np.average(mse2[0]))
        print('error at 100 step:', np.average(mse2[-1]))



        horizon = 2000
        Y_gen = np.zeros((U.shape[0], horizon))
        for h in range(horizon):
            if h == 0:
                U_test = U[:, test_start: test_start + 1]
                X_test = X[:, test_start: test_start + 1]
            else:
                U_test = Y_pred
                X_test = (1 - a) * X_test + a * np.tanh(Win @ np.vstack((np.ones((1, 1)), U_test)) + W @ X_test)
            H = np.vstack((np.ones((1, 1)), X_test))
            Y_pred = Wout @ H
            # print(Y_gen[:, h] ,Y_pred)
            Y_gen[:, h:h+1] = Y_pred

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*U[:3, :2000])
        ax.plot(*Y_gen[:3, :2000])
        plt.show()

        ######################### ATTN #######################

        # Ml, Diag, Mr = np.linalg.svd(X_train, full_matrices=False)
        # tau = svht(X_train, sv=Diag)
        # N_v = np.sum(Diag > tau)
        # MlT, Diag, Mr = Ml.T[:N_v], Diag[:N_v], Mr[:N_v]
        #
        # # Cs3 = (np.diag(Diag) @ Mr).T[np.random.choice(num_train, N_c,replace=False)]
        #
        # Cs3 = clustering((np.diag(Diag) @ Mr).T, N_c)  # N_c, N_v
        #
        # #     Cs3 = np.random.random([N_c, N_v])
        # #     okm = OnlineKMeans(N_v, N_c, flag=True, random_init=False)
        # #     for p in tqdm((MlT@X_train).T):
        # #         okm.fit(p)
        # #     Cs3 = okm.centroid
        #
        # choices = []
        # for beta in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]:
        #     S = np.exp(- beta * pairwise_distances(Cs3, (np.diag(Diag) @ Mr).T))
        #     # S_trans = S[:,1:] @ S[:,:-1].T @ np.linalg.inv(S[:,:-1] @ S[:,:-1].T + reg * np.eye(N_c))
        #     # S = S_trans @ S
        #     H = np.vstack((np.ones((1, num_train)), X_train, S))
        #     Wout3 = Y_train @ H.T @ np.linalg.inv(H @ H.T + reg * np.eye(H.shape[0]))
        #
        #     # validation
        #     horizon = 100
        #     mse3 = []
        #     for h in range(horizon):
        #         if h == 0:
        #             U_val = U[:, val_start: val_start + num_val]
        #             X_val = X[:, val_start: val_start + num_val]
        #         else:
        #             U_val = Y_pred
        #             X_val = (1 - a) * X_val + a * np.tanh(Win @ np.vstack((np.ones((1, num_val)), U_val)) + W @ X_val)
        #         S = np.exp(- beta * pairwise_distances(Cs3, (MlT @ X_val).T))
        #         H = np.vstack((np.ones((1, num_val)), X_val, S))
        #         Y_pred = Wout3 @ H
        #         Y_true = U[:, val_start + h + 1: val_start + num_val + h + 1]
        #         mse3.append(np.average(np.sum(np.square(Y_pred - Y_true), axis=0)))
        #     choices.append((mse3[-1], beta, Wout3))
        # train_error, beta3, Wout3 = sorted(choices)[0]
        #
        # # test
        #
        # horizon = 100
        # mse3 = []
        # for h in range(horizon):
        #     if h == 0:
        #         U_test = U[:, test_start: test_start + num_test]
        #         X_test = X[:, test_start: test_start + num_test]
        #     else:
        #         U_test = Y_pred
        #         X_test = (1 - a) * X_test + a * np.tanh(Win @ np.vstack((np.ones((1, num_test)), U_test)) + W @ X_test)
        #     S = np.exp(- beta3 * pairwise_distances(Cs3, (MlT @ X_test).T))
        #     H = np.vstack((np.ones((1, num_test)), X_test, S))
        #     Y_pred = Wout3 @ H
        #     Y_true = U[:, test_start + h + 1: test_start + num_test + h + 1]
        #     # Y_pred1 = Ml1[-1] @ Y_pred
        #     # Y_true = U0[:, test_start + h + 9: test_start + num_test + h + 9]
        #     mse3.append(np.sum(np.square(Y_pred - Y_true), axis=0))
        #
        # print('error at   1 step:', np.average(mse3[0]))
        # print('error at 100 step:', np.average(mse3[-1]))
        #
        # # horizon = 2000
        # # Y_gen = np.zeros((U.shape[0], horizon))
        # # for h in range(horizon):
        # #     if h == 0:
        # #         U_test = U[:, test_start: test_start + 1]
        # #         X_test = X[:, test_start: test_start + 1]
        # #     else:
        # #         U_test = Y_pred
        # #         X_test = (1 - a) * X_test + a * np.tanh(Win @ np.vstack((np.ones((1, 1)), U_test)) + W @ X_test)
        # #     S = np.exp(- beta3 * pairwise_distances(Cs3, (MlT @ X_test).T))
        # #     H = np.vstack((np.ones((1, 1)), X_test, S))
        # #     Y_pred = Wout3 @ H
        # #     Y_gen[:, h:h + 1] = Y_pred
        # #
        # # fig = plt.figure(figsize=(7, 7))
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.plot(*U[:, :2000])
        # # ax.plot(*Y_gen[:2000])
        # # plt.show()



