# setupPly.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import open3d as o3d
import argparse
import os
from glob import glob
import pickle
from global_registration import preprocess_point_cloud, execute_global_registration
import copy


import numpy as np
import time


def arguments_parser():

    parser = argparse.ArgumentParser()

    # parser.add_argument("--file_name", type=str, default='data.pickle', help="Please write a filename")
    parser.add_argument("--dataset", type=str, default='sun3d', help="Please write the folders name - dataset")
    parser.add_argument("--dataset_dir", type=str, default='test', help="Please write the folders name - dataset_dir")

    return parser.parse_args()

def getPcRGBD(path, idx):

    frame = 'frame-000000'
    frame = frame[:-len(idx)] + idx
    depth = frame + '.depth.png'
    depth = os.path.join(path, depth)
    rgb = frame + '.color.png'
    rgb = os.path.join(path, rgb)


    im_depth = o3d.read_image(depth)
    im_color = o3d.read_image(rgb)

    # rgbd_image = o3d.create_rgbd_image_from_color_and_depth(im_color, im_depth)

    pcd = o3d.create_point_cloud_from_depth_image(im_depth, o3d.PinholeCameraIntrinsic(
    # pcd = o3d.create_point_cloud_from_rgbd_image(rgbd_image, o3d.PinholeCameraIntrinsic(
        o3d.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # o3d.draw_geometries([pcd])

    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source.paint_uniform_color([1, 0, 0])
    source_temp.transform(transformation)

    vis = o3d.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(source_temp)
    pos = os.path.join(os.getcwd(), 'registration', 'pos.json')
    vis.get_render_option().load_from_json(pos)
    vis.run()  # user picks points
    vis.destroy_window()

    # o3d.draw_geometries([source_temp, target_temp])

def RotationError(R1, R2):

    R1 = np.real(R1)
    R2 = np.real(R2)

    R_ = np.matmul(R1, np.linalg.inv(R2))

    ae = np.arccos((np.trace(R_) - 1)/2)
    ae = np.rad2deg(ae)

    frob_norm = np.linalg.norm(R_ - np.eye(3), ord='fro')

    return ae, frob_norm

def TranslationError(t1, t2):

    return np.linalg.norm(t1-t2)


def refine_registration(source, target, corrs):


    start = time.time()

    p2p = o3d.TransformationEstimationPointToPoint()
    result = p2p.compute_transformation(source, target, o3d.Vector2iVector(corrs))

    elapsed_time = time.time() - start

    return result, elapsed_time


def initializePointCoud(pts):

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(pts)

    return pcd

def makeMatchesSet(corres):

    set = []

    for idx, c in enumerate(corres):

        if c == 1:
            set.append([idx, idx])

    return set

def createPointsfromMatches(pc1, pc2, matches):

    x1_ = np.asarray(pc1.points)
    x2_ = np.asarray(pc2.points)
    x1 = []
    x2 = []

    for match in matches:
        x1.append(x1_[match[0]])
        x2.append(x2_[match[1]])

    x1 = np.array(x1)
    x2 = np.array(x2)

    ones = np.ones((matches.shape[0], 1))
    flag = ones.reshape(matches.shape[0])
    out = np.arange(0, matches.shape[0])
    np.random.shuffle(out)
    nb_out = int(np.random.uniform(low=0.45, high=0.55) * matches.shape[0])
    out = out[:nb_out]
    flag[out] = 0
    noise = np.random.normal(0, 0.1, size=(nb_out, 3))
    x1_noise = x1
    x1_noise[out] = x1_noise[out] + noise
    flag = np.array(flag, dtype=int)

    return x1_, x2_, flag



def main():

    voxel_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    # voxel_list = [ 0.05, 0.04, 0.03]

    prefix = '*.pickle'
    # voxel_size = 0.06

    config = arguments_parser()

    filepath = os.path.join(os.getcwd(), 'data', config.dataset, config.dataset_dir, prefix)
    # filepath = os.path.join(os.getcwd(), '..', 'data', config.dataset, prefix)
    filepath1 = os.path.join(os.getcwd(), 'registration', prefix)
    # filepath = os.path.abspath(filepath)
    files1 = glob(filepath1)
    files = glob(filepath)

    # angles = np.random.random_integers(-90,90,100)

    f = open(files[1], 'rb')
    data = pickle.load(f)
    f.close()

    s_idx1 = '{}'.format(data['idx1'][66].reshape(1)[0])
    s_idx2 = '{}'.format(data['idx2'][66].reshape(1)[0])

    pc1 = getPcRGBD(os.path.join(os.getcwd(), 'registration'), s_idx1)
    pc2 = getPcRGBD(os.path.join(os.getcwd(), 'registration'), s_idx2)
    print(pc1)

    minCorres = False
    minMatchNb = 0
    minpc1 = np.array([])
    minpc2 = np.array([])
    minMatches = np.array([])

    for voxel_size in voxel_list:

        pc1_down, pc1_fpfh = preprocess_point_cloud(pc1, voxel_size)
        pc2_down, pc2_fpfh = preprocess_point_cloud(pc2, voxel_size)

        T = np.eye(4)
        T[:3, :3] = data['R'][66]
        T[:3, 3] = data['t'][66]

        star_t = time.time()
        result = execute_global_registration(pc1_down, pc2_down, pc1_fpfh, pc2_fpfh, voxel_size)
        delta = time.time() - star_t

        print('RANSAC')
        print('Delta = {}'.format(delta))
        rot_error, frob_error = RotationError(T[:3, :3], result.transformation[:3, :3])
        print('Rotation Error: {} deg / {} (frob)\nTranslation Error: {}'.format(np.abs(rot_error), frob_error,
                                                                                 np.linalg.norm(T[:3, 3] - result.transformation[:3,3])))


        matches = np.asarray(result.correspondence_set)
        print('Number of Correspondences = {}'.format(matches.shape[0]))

        if matches.shape[0] > 2700 and matches.shape[0] < 3500:
            # print('Number of Correspondences = {}'.format(matches.shape[0]))

            x1, x2, flag = createPointsfromMatches(pc1_down, pc2_down, matches)
            print('Matches set created')

            pts1 = initializePointCoud(x1)
            pts2 = initializePointCoud(x2)

            pc1_down, pc1_fpfh = preprocess_point_cloud(pts1, voxel_size)
            pc2_down, pc2_fpfh = preprocess_point_cloud(pts2, voxel_size)

            result = execute_global_registration(pc1_down, pc2_down, pc1_fpfh, pc2_fpfh, voxel_size)
            matches = np.asarray(result.correspondence_set)
            print('Number of Correspondences = {}'.format(matches.shape[0]))

            rot_error, frob_error = RotationError(T[:3, :3], result.transformation[:3, :3])
            print('Rotation Error: {} deg / {} (frob)\nTranslation Error: {}'.format(np.abs(rot_error), frob_error,
                                                                                     np.linalg.norm(T[:3,
                                                                                                    3] - result.transformation[
                                                                                                         :3, 3])))

            minCorres = True
            break

            # x1_ = np.asarray(pc1_down.points)
            # x2_ = np.asarray(pc2_down.points)
            # x1 = []
            # x2 = []
            #
            # for match in matches:
            #
            #     x1.append(x1_[match[0]])
            #     x2.append(x2_[match[1]])
            #
            # x1 = np.array(x1)
            # x2 = np.array(x2)

            # # Não interessa
            # ones = np.ones((matches.shape[0], 1))
            # x1_h = np.concatenate([x1, ones], axis = 1)
            # err = np.matmul(T,np.transpose(x1_h))
            # err = x2 - np.transpose(err[:3])
            # err = np.linalg.norm(err, axis=1)
            #
            #
            #
            #
            # # Não interessa
            # x1_h = np.concatenate([x1_noise, ones], axis=1)
            # err = np.matmul(T, np.transpose(x1_h))
            # err = x2 - np.transpose(err[:3])
            # err = np.linalg.norm(err, axis=1)
            # flag2 = ones.reshape(matches.shape[0])
            # flag2[err > 0.3] = 0
            #
            # print(np.sum(flag)/matches.shape[0])
            # print(np.sum(flag2)/matches.shape[0])
            # print(err)

        else:

            if minMatchNb < matches.shape[0] and matches.shape[0] > 3500:

                minpc1 = pc1_down
                minpc2 = pc2_down

                minMatches = matches


    if np.array(minpc1.points).size > 0 and not minCorres:

        if not minCorres:

            x1, x2, flag = createPointsfromMatches(minpc1, minpc2, minMatches[:3000])
            print('Matches set created')
            print('Number of Correspondences = {}'.format(minMatches[:3000].shape[0]))

    else:
        print('Discard Frame')



        # print('Number of Correspondences = {}'.format(matches.shape[0]))
        # rot_error, frob_error = RotationError(T[:3, :3], result.transformation[:3, :3])
        # print('Rotation Error: {} deg / {} (frob)\nTranslation Error: {}'.format(np.abs(rot_error),frob_error,
        #     np.linalg.norm(T[:3, 3] - result.transformation[:3, 3])))
        #
        # print('Umeyama')
        # transformation, elapsed = refine_registration(pc1_down, pc2_down, result.correspondence_set)
        # rot_error, frob_error = RotationError(T[:3, :3], transformation[:3, :3])
        # print('Rotation Error: {} deg / {} (frob)\nTranslation Error: {}'.format(np.abs(rot_error), frob_error,
        #                                                                          np.linalg.norm(T[:3, 3] - transformation[:3,3])))
        #
        # print('Delta = {}'.format(elapsed))
        # print('Total Delta = {}'.format(delta+elapsed))



if __name__ == '__main__':

    main()