# data.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import os
import pickle
from glob import glob
import numpy as np
from transformations import rotation_matrix
from ops import np_matrix4_vector_mul


def load_data(config, mode):

    print('Loading %s data' % mode)
    # prefix = 'data.pickle'
    prefix = '*.pickle'

    data_names = getattr(config, "data_dir_" + mode[:2])
    data = {}

    for name in data_names:

        # name += '_3views'
        filepath = os.path.join(os.getcwd(), config.data_pre, name, mode, prefix)

        files = glob(filepath)
        if not files:
            # data_gen_lock.unlock()
            raise RuntimeError("Data is not prepared!")


        for filename in files:

            pos_point = [i for i, letter in enumerate(filename) if letter == '.'][-1]
            type = filename[pos_point + 1:]

            if type != 'pickle':
                continue

            file = open(filename, 'rb')
            data_temp = pickle.load(file)
            file.close()

            if not data:
                data = data_temp

            else:
                for key in data:
                    data[key] += data_temp[key]

            # else:
            #     for key in data:
            #         if key != 'trios':
            #             for k in data[key]:
            #                 data[key][k] += data_temp[key][k]
            #         else:
            #             data[key] += data_temp[key]


    return data


def separateBatch(x, b):
    return [x[i] for i in b]


def prepareBatch(data, batch_size, max_steps):

    batches = np.arange(0, len(data['x1']))
    np.random.shuffle(batches)
    batches = np.array_split(batches[:int(batch_size*max_steps)], max_steps)

    x1 = data['x1']
    x2 = data['x2']
    Rs = data['R']
    ts = data['t']
    fs = data['flag']

    x1_b = []
    x2_b = []
    Rs_b = []
    ts_b = []
    fs_b = []

    for b in batches:

        x1_b.append(separateBatch(x1, b))
        x2_b.append(separateBatch(x2, b))
        Rs_b.append(separateBatch(Rs, b))
        fs_b.append(separateBatch(fs, b))
        ts_b.append(separateBatch(ts, b))


    return x1_b, x2_b, Rs_b, ts_b, fs_b

def computeAngle(epoch):

    return 2*(epoch % 25)


def dataTransform(x1_b, Rs_b, ts_b, epoch, aug_cl = False):

    angle = computeAngle(epoch)

    x1 = x1_b
    R = Rs_b
    t = ts_b

    x1_b = []
    Rs_b = []
    ts_b = []

    step = epoch

    for i in range(len(x1)):

        x1_b1 = []
        R_b1 = []
        t_b1 = []

        if not aug_cl:
            angle = computeAngle(step)

        for j in range(len(x1[i])):

            d = np.random.rand(3)
            d = d / np.linalg.norm(d)
            T = rotation_matrix(np.deg2rad(angle), d)

            ones = np.ones(shape=(x1[i][j].shape[0], 1))
            x1_o = np.concatenate((x1[i][j], ones), axis=1)
            x1_o = np_matrix4_vector_mul(np.transpose(T), x1_o)

            T_gt = np.eye(4)
            T_gt[:3,:3] = R[i][j]
            T_gt[:3, 3] = t[i][j]
            T = np.matmul(T_gt, T)

            x1_b1.append(x1_o[:,:3])
            R_b1.append(T[:3,:3])
            t_b1.append(T[:3, 3])

        x1_b.append(x1_b1)
        Rs_b.append(R_b1)
        ts_b.append(t_b1)

        step += 1

    return x1_b, Rs_b, ts_b
