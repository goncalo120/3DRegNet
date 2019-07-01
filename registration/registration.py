# registration.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import open3d as o3
# from global_registration import *
from registration.global_registration import *
import time

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            FastGlobalRegistrationOption(maximum_correspondence_distance = distance_threshold))

    return result

def refine_registration(source, target, corrs):


    start = time.time()

    p2p = o3.TransformationEstimationPointToPoint()
    result = p2p.compute_transformation(source, target, o3.Vector2iVector(corrs))

    elapsed_time = time.time() - start

    return result, elapsed_time

def initializePointCoud(pts):

    pcd = o3.PointCloud()
    pcd.points = o3.Vector3dVector(pts)

    return pcd

def preProcessData(pts, flag):

    return initializePointCoud(pts[flag])
    # return initializePointCoud(pts)

def selectFunction(name):

    if name == 'global':
        return execute_global_registration

    elif name == 'fast':
        return execute_fast_global_registration

    else:
        raise ValueError('Please select a valid function: global or fast')



def globalRegistration(source, target, execute_registration):

    voxel_size = 0.05

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # print(source_down)
    # print(target_down)

    voxel_size = 0.05
    start = time.time()
    result = execute_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    elapsed_time = time.time() - start

    return result, elapsed_time, source_down, target_down


def makeMatchesSet(corres):
    set = []

    corres = corres.reshape(corres.shape[1])
    # print(corres.shape)

    for idx, c in enumerate(corres):

        if c == 1:
            set.append([idx, idx])

    return set
