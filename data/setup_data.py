# config.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import argparse
from readMat import readPickle
import os
import sys
import pickle
from glob import glob
from numpy import mean


def arguments_parser():

    parser = argparse.ArgumentParser()

    # parser.add_argument("--file_type", type=str, default='mat', choices=['mat', 'h5'],
    #                     help="choose which file type: mat, h5")

    # parser.add_argument("--file_name", type=str, default='data.mat', help="Please write a filename")
    parser.add_argument("--dataset_dir", type=str, default='sun3d', help="Please write the folders name")
    parser.add_argument("--mode", type=str, default='train', help="Please write the mode")

    return parser.parse_args()

def check_args(config):

    type_list = ['pickle']

    filepath = os.path.join(os.getcwd(), config.dataset_dir, config.mode, '*.pickle')

    files = glob(filepath)
    if not files:
        # data_gen_lock.unlock()
        raise RuntimeError("Data is not prepared!")

    newfiles = []
    types = []

    for filename in files:
        pos_point = [i for i, letter in enumerate(filename) if letter == '.'][-1]
        newfile = filename[:pos_point] +'.pickle'
        newfile = os.path.join(os.getcwd(), config.dataset_dir, config.mode, newfile)
        newfiles.append(newfile)

        type = filename[pos_point+1:]

        if not type in type_list:
            print('Error: Type not in list')
            exit(2)

        types.append(type)

    return types, files, newfiles


def dataToPickle(file, data):

    f = open(file, 'wb')
    pickle.dump(data, f)
    f.close()


def dataToh5():
    pass

def main():

    config = arguments_parser()
    types, files, newfiles = check_args(config)

    print(sys.path)

    data = []
    length = 0

    per = []

    for idx, filename in enumerate(files):

        if types[idx] == 'pickle':

            mode = config.mode[0:2]

            if os.path.exists(config.dataset_dir):
                folder = os.path.join(config.dataset_dir, config.mode)
                data, mper = readPickle(filename, folder)
                per.append(mper)

            else:
                data, mper = readPickle(filename, 'unprocessed', mode)
                per.append(mper)

            length += len(data['R'])

            dataToPickle(newfiles[idx], data)

        else:
            print('Error: Type not in list')

    print('Percentage of Inliers per dataset:')
    for idx, filename in enumerate(files):
        print('{} - {}'.format(filename, per[idx]))

    print('Mean: {}'.format(mean(per)))


    print('Total files for {} =  {}'.format(config.mode, length))



if __name__ == '__main__':

    main()
