from RBFNN import RBFNN
from dataset.chaotic_system import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
import copy
import os
import time





class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


names = [
    # 'rossler',
    'rabinovich_fabrikant',
    # 'lorentz',
    # 'chen',
    # 'chua',
    # 'switch'
]
names = [n + '1d' for n in names]

n_dim = 1
horizon = 1
N = 10000
n_history = 1  # 使用 n 个历史点作为输入
N_h = 200
num_train = 2000
train_start = 0
num_test = 2000
test_start = 5000
np.random.seed()
nz = 100
connectivity = 1
reservoirConf = Dict(
    alpha=0.9,
    connectivity= connectivity,
    nz= nz,
    nu = n_dim,
    target_rho=0.99,
    input_scale= 1
)

for file in range(50):
    result_file = './result_1d_rabinovich_fabrikant'+str(file) + '.csv'
    if os.path.exists(result_file):
        result = pd.read_csv(result_file)
    else:
        result = pd.DataFrame(columns=['datetime', 'system_name',  'n_dim', 'n_history', 'horizon', 'model_name', 'N_h', 'skip_con', 'sigma','reservoir_encoder','nz', 'connectivity', 'mse'])

    process_start_time = time.time()
    print('开始数据处理')

    for n_history in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:
    # for n_history in [300, 500]:
        for system_name in names:

            print(system_name)
            '''
            数据集
            '''
            x = np.loadtxt('dataset/'+system_name+'.txt', delimiter=',').T
            x += np.random.randn(*x.shape)*0.001
            # x = np.array([[0.3, 0.3, 0.4]]) @ x

            def show_dateset(x):
                print(x.shape)
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plt.plot(*x, 'g')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(system_name + ' system')
                # plt.savefig('../figures/' + system_name + '_system.pdf')
                # plt.show()

                plt.figure(figsize=(20, 10))
                plt.title(system_name)
                dim = ['x', 'y', 'z']
                for i in range(n_dim):
                    plt.subplot(n_dim, 1, i + 1)
                    plt.plot(x[i], color='g')
                    plt.ylabel(dim[i])
                    plt.legend(loc='upper right')
                plt.xlabel('t')
                # plt.show()

            x_train = np.vstack([select_samples(x, train_start + i, num_train) for i in range(n_history)])
            x_test = np.vstack([select_samples(x, test_start + i, num_test) for i in range(n_history)])
            y_test = select_samples(x, test_start + n_history + horizon - 1, num_test)
            y_train = select_samples(x, train_start + n_history + horizon - 1, num_train)

            rc = reservoirConf

            # print('train set:', x_train.shape, y_train.shape)
            # print('test  set:', x_test.shape, y_test.shape)


            def show_train_test():
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plt.plot(*x_train[:3, :], 'green', label='train set')
                plt.plot(*y_test, 'coral', label='test set')
                # plt.plot(*model.W_i[:,:3].T, 'ko',label='hidden layer')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.title(system_name)
                # plt.savefig('../figures/' + system_name + '_split.pdf')
                plt.show()
            # show_train_test()


            def show_train():
                plt.figure(figsize=(40, 10))
                dim = ['x', 'y', 'z']
                for i in range(3):
                    plt.subplot(3, 1, i + 1)
                    plt.plot(y_train[i, :].T, color='green', label='train set')
                    plt.ylabel(dim[i])
                    plt.legend(loc='upper right')
                plt.xlabel('t')
                plt.savefig('../figures/'+system_name+'.png')
                plt.show()
            # show_train()


            '''
            单步预测
            '''
            # (model_name, n_neuron, skip_con, , sigma, rc, type, use_state )
            model_confs = []
            model_confs += [('ESN', 0, 1, 1, rc, 2, True)]
            model_confs += [('RBFLN RE 1', N_h, 1, sigma, rc, 1, False) for sigma in
                            [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]]
            # model_confs += [('RBFLN RE 2', N_h, 1, sigma, rc, 2, False) for sigma in
            #                 [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]]
            model_confs += [('RBFLN', N_h, 1, sigma, None, 1, False) for sigma in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]]
            model_confs += [('VAR', 0, 1, 1, None, 1, False) ]
            model_confs += [('RBFN', N_h, 0, sigma, None, 1, False) for sigma in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]]


            model_names = [conf[0] for conf in model_confs]
            # colors = ['red', 'blue', 'purple', 'green', 'orange',  'pink', '']
            colors = list(cnames.keys())

            def single_step_experiment():
                Predictions = [np.empty((n_dim, num_test)) for _ in range(len(model_confs))]
                MSE = [0.0] * len(model_confs)

                for j, conf in enumerate(model_confs):
                    model_name, n_neuron, skip_con, sigma, rc, encoder_type, use_reseroir_state = conf
                    conf_dict = dict(
                        N_h=n_neuron,
                        skip_con=skip_con,
                        sigma=sigma,
                        reservoirConf=rc,
                        encoder_type=encoder_type,
                        use_reseroir_state=use_reseroir_state
                    )

                    model = RBFNN(**conf_dict)

                    model.train(x_train, y_train)

                    Predictions[j] = model.predict(x_test)

                    MSE[j] = mse(Predictions[j], y_test)

                    result.loc[result.shape[0]] = {
                        'datetime': pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'system_name': system_name,
                       'n_dim' : n_dim,
                        'n_history' : n_history,
                        'horizon' : horizon,
                        'model_name' : model_name,
                        'N_h' : N_h,
                        'sigma' : sigma,
                        'nz' : nz,
                        'connectivity' : connectivity,
                       'skip_con': skip_con,
                       'reservoir_encoder': 0 if rc is None else 1,
                       'mse': MSE[j]
                    }
                    print(result.loc[result.shape[0]-1])

                # plt.figure()
                # plt.bar(model_names, MSE)
                #
                # for j in range(len(model_confs)):
                #     print(model_names[j], MSE[j])

                # plt.figure(figsize=(40, 10))
                # plt.title(system_name)
                # dim = ['x', 'y', 'z']
                # for i in range(n_dim):
                #     plt.subplot(n_dim, 1, i + 1)
                #     for j in range(len(model_confs)):
                #         plt.plot(Predictions[j][i, :].T, color=colors[j], label=model_names[j])
                #     plt.plot(y_test[i, :].T, '--k', label='ground truth')
                #     plt.ylabel(dim[i])
                #     plt.legend(loc='upper right')
                # plt.xlabel('t')
                # plt.show()

            single_step_experiment()


    result.to_csv(result_file, index=False)

    print('结束数据处理')
    process_stop_time = time.time()

    # 差的时间戳
    diff_time = process_stop_time - process_start_time
    # 将计算出来的时间戳转换为结构化时间
    struct_time = time.gmtime(diff_time)
    # 减去时间戳最开始的时间 并格式化输出
    print('数据处理用了{0}年{1}月{2}日{3}小时{4}分钟{5}秒'.format(
        struct_time.tm_year - 1970,
        struct_time.tm_mon - 1,
        struct_time.tm_mday - 1,
        struct_time.tm_hour,
        struct_time.tm_min,
        struct_time.tm_sec
    ))