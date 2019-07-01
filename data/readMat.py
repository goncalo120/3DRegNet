# config.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import scipy.io as sio
import os
import numpy as np
import pickle

def readMat(filename, folder, mode):

    filename = os.path.join(os.getcwd(), os.path.join(folder, filename))
    print(filename)

    f = sio.loadmat(filename)
    # import h5py
    # with h5py.File(filename, 'r') as f:
    #
    #     items = f['data']
    #     print(list(items.keys()))


    # f = f[mode[:2]][0]
    f = f['data'][0]

    R = []
    t = []
    x1_list = []
    x2_list = []
    flag = []
    idx1 = []
    idx2 = []


    for item in f:

        R.append(item['T'][0:3,0:3])
        t.append(item['T'][0:3,3])
        # x1 = item['x1']
        # x1 = x1/10
        x1_list.append(item['x1'])
        x2_list.append(item['x2'])
        fl = np.array(item['flag'])
        flag.append(fl.reshape(fl.shape[0]))
        idx1.append(item['idx1'])
        idx2.append(item['idx2'])

    data = {}

    data['R'] = R
    data['t'] = t
    data['x1'] = x1_list
    data['x2'] = x2_list
    data['flag'] = flag
    data['idx1'] = idx1
    data['idx2'] = idx2

    return data

def createFlags(x1, x2, T):

    ones = np.ones((x1.shape[0], 1))
    flag = ones.reshape(x1.shape[0])

    error = 0.1

    x1_h = np.concatenate([x1, ones], axis=1)
    err = np.matmul(T, np.transpose(x1_h))
    err = x2 - np.transpose(err[:3])
    err = np.linalg.norm(err, axis=1)
    flag[err > error] = 0

    print('Number of Inliers = {}\tPercentage = {}'.format(np.sum(flag), np.sum(flag) / x1.shape[0]))

    return np.array(flag, dtype=int), np.sum(flag) / x1.shape[0]




def readPickle(filename, folder):

    filename = os.path.join(os.getcwd(), os.path.join(folder, filename))
    print(filename)

    f = open(filename, 'rb')
    data_original = pickle.load(f)
    f.close()

    R = []
    t = []
    x1_list = []
    x2_list = []
    flag = []
    idx1 = []
    idx2 = []
    ransacT = []
    ransacUT = []
    fgrT = []
    ransacDelta = []
    ransacUDelta = []
    fgrDelta = []

    per = []

    for frame in data_original:

        if frame['flag'].size != 0:
            R.append(frame['R'])
            t.append(frame['t'])
            x1_list.append(frame['x1'])
            x2_list.append(frame['x2'])
            T = np.eye(4)
            T[:3,:3] = frame['R']
            T[:3,3] = frame['t']
            f, p = createFlags(frame['x1'], frame['x2'], T)
            per.append(p)
            flag.append(f)
            idx1.append(frame['idx1'])
            idx2.append(frame['idx2'])
            ransacT.append(frame['ransacT'])
            ransacUT.append(frame['ransacUT'])
            fgrT.append(frame['ransacUT'])
            ransacDelta.append(frame['ransacDelta'])
            ransacUDelta.append(frame['ransacUDelta'])
            fgrDelta.append(frame['fgrDelta'])


    print(np.mean(per))

    data = {}

    data['R'] = R
    data['t'] = t
    data['x1'] = x1_list
    data['x2'] = x2_list
    data['flag'] = flag
    data['idx1'] = idx1
    data['idx2'] = idx2
    data['ransacT'] = ransacT
    data['ransacUT'] = ransacUT
    data['fgrT'] = fgrT
    data['ransacDelta'] = ransacDelta
    data['ransacUDelta'] = ransacUDelta
    data['fgrDelta'] = fgrDelta

    return data, np.mean(per)