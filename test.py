from RBFNN import RBFNN
from chaotic_system import gen_model, trajectory
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rescale(x):
    row_max = np.max(x, axis=1).reshape((x.shape[0], 1))
    row_min = np.min(x, axis=1).reshape((x.shape[0], 1))
    return (x - row_min) / (row_max - row_min)

def select_samples(X, start, num):
    return X[:, start:start + num]


names = ['rossler', 'rabinovich_fabrikant', 'lorentz', 'chen', 'chua', 'switch']
system_name = names[1]
horizon = 30
N = 10000
n = 10  # 使用 n 个历史点作为输入
num_train = 5000
train_start = 0
num_test = 2000
test_start = 6000

'''
数据集
'''

functions, start_point, step = gen_model(system_name)
x = trajectory(functions, start_point, N, step)
x = rescale(x)

# pertubed = [i + np.random.randn() * 0.01 for i in start_point]
# x_ = trajectory(functions, pertubed, N, step)
# x_ = rescale(x_)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(*x, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title(system_name + ' system')
plt.savefig('../figures/ ' + system_name + '_system.pdf')

plt.show()
plt.figure(figsize=(20, 10))
plt.title(system_name)
dim = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(x[i, :].T, color='g')
    plt.ylabel(dim[i])
    plt.legend(loc='upper right')
plt.xlabel('t')
plt.show()


x_train = np.vstack([select_samples(x, train_start + i, num_train) for i in range(n)])
y_train = select_samples(x, train_start + n, num_train)

x_test = np.vstack([select_samples(x, test_start + i, num_test) for i in range(n)])  # test set
y_test = select_samples(x, test_start + n, num_test)

print('train set:', x_train.shape, y_train.shape)
print('test  set:', x_test.shape, y_test.shape)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(*x_train[:3, :], 'green', label='train set')
plt.plot(*y_test, 'coral', label='test set')
# plt.plot(*model.W_i[:,:3].T, 'ko',label='hidden layer')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(system_name)
plt.savefig('../figures/ ' + system_name + '_split.pdf')
plt.show()

plt.figure(figsize=(40, 10))
dim = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(y_train[i, :].T, color='green', label='train set')
    plt.ylabel(dim[i])
    plt.legend(loc='upper right')
plt.xlabel('t')

'''
单步预测
'''
np.random.seed(42)
Predictions = [np.empty((3, num_test)) for _ in range(4)]
use_pred = False

model_confs = [('RBFN', 100, 0, 0), ('VAR', 0, 1, 0), ('FLNN', 100, 1, 0), ('RBFAR', 100, 1, 1)]

for j, conf in enumerate(model_confs):
    model_name, n_neuron, skip_con, reweight = conf

    model = RBFNN(n_neuron,
                  skip_con=skip_con,
                  reweight=reweight
                  )

    model.train(x_train, y_train)
    p = x_test[:, :1]
    for i in range(num_test):
        p_next = model.predict(p)
        Predictions[j][:, i] = np.squeeze(p_next)
        if use_pred:
            p = np.vstack([p, p_next])[3:]
        else:
            p = x_test[:, i + 1: i + 2]


model_names = ['RBFN', 'VAR', 'FLNN', 'RBFAR']
colors = ['red', 'blue', 'purple', 'green']

plt.figure(figsize=(40, 10))
plt.title(system_name)
dim = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    for j in range(4):
        plt.plot(Predictions[j][i, :].T, color=colors[j], label=model_names[j])
    plt.plot(y_test[i, :].T, color='black', label='ground truth')
    plt.ylabel(dim[i])
    plt.legend(loc='upper right')
plt.xlabel('t')
plt.show()


'''
多步预测
'''


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


def error_multistep(err_func, A, B):
    """
    A, B: np.array with shape (dim * horizon, 1)
    """
    assert (A.shape == B.shape)
    dim = 3
    horizon = A.shape[0] // dim
    mse_at_each_horizon = []
    for i in range(horizon):
        mse_at_each_horizon.append(err_func(A[i * dim:(i + 1) * dim], B[i * dim:(i + 1) * dim]))
    return mse_at_each_horizon


fig, ax = plt.subplots(figsize=(6, 6))
for conf, color in zip(model_confs, colors):
    model_name, n_neuron, skip_con, reweight = conf

    model = RBFNN(n_neuron,
                  skip_con=skip_con,
                  reweight=reweight
                  )

    model.train(x_train, y_train)

    Y_mutistep = model.predict_multistep(x_test, horizon)
    Y_true = np.vstack([select_samples(x, test_start + n + i, num_test) for i in range(horizon)])

    mean_list = []
    for i in range(num_test):
        A = Y_mutistep[:, i:i + 1]
        B = Y_true[:, i:i + 1]
        err_list = error_multistep(mse, A, B)
        mean_list.append(err_list)

    mean_list = np.average(mean_list, axis=0)
    plt.plot([i + 1 for i in range(horizon)], mean_list, label=model_name, color=color)
    plt.ylabel('mse')
    plt.xlabel('forecast horizon')
#     plt.grid(True)

ax.set_yscale("log")
plt.legend(loc='lower right')
plt.title(system_name)
plt.savefig('../figures/' + system_name + '.pdf')
plt.show()

'''
部分样本
'''


# model_names = ['RBFN','VAR','FLNN','RBFAR']
# model_name, n_neuron, skip_con, reweight = model_confs[3]

# model = RBFNN(n_neuron,
#               skip_con=skip_con,
#               reweight=reweight
#              )

# model.train(x_train, y_train)
# Y_mutistep = model.predict_mutistep(x_test,horizon)
# Y_true = np.vstack([select_samples(x, test_start+n+i, num_test) for i in range(horizon)])

# for i in [50,200,500,1000]:
#     A = Y_mutistep[:,i:i+1]
#     B = Y_true[:,i:i+1]
#     plt.figure()
#     for j in range(3):
#         plt.subplot(3,1,j+1)
#         plt.plot(A[j::3], label='predict')
#         plt.plot(B[j::3], label='true')
#         plt.legend(loc='upper right')
