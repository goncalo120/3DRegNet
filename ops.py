# ops.py ---
#
# Filename: ops.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:  
# Created: Tue Apr  3 14:09:17 2018 (-0700)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Modified by: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
#  Instituto Superior Técnico (IST)

# Change Log:
# 
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import numpy as np
from tensorflow.python.framework import function
import tensorflow as tf

from six.moves import xrange


# ------------------------------------------------------------
# Tensorflow ops

def tf_get_shape_as_list(x):

    return [_s if _s is not None else - 1 for _s in x.get_shape().as_list()]


def tf_quaternion_from_matrix(M):

    import tensorflow as tf

    m00 = M[:, 0, 0][..., None]
    m01 = M[:, 0, 1][..., None]
    m02 = M[:, 0, 2][..., None]
    m10 = M[:, 1, 0][..., None]
    m11 = M[:, 1, 1][..., None]
    m12 = M[:, 1, 2][..., None]
    m20 = M[:, 2, 0][..., None]
    m21 = M[:, 2, 1][..., None]
    m22 = M[:, 2, 2][..., None]
    # symmetric matrix K
    zeros = tf.zeros_like(m00)
    K = tf.concat(
        [m00 - m11 - m22, zeros, zeros, zeros,
         m01 + m10, m11 - m00 - m22, zeros, zeros,
         m02 + m20, m12 + m21, m22 - m00 - m11, zeros,
         m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        axis=1)
    K = tf.reshape(K, (-1, 4, 4))
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = tf.self_adjoint_eig(K)

    q0 = V[:, 3, 3][..., None]
    q1 = V[:, 0, 3][..., None]
    q2 = V[:, 1, 3][..., None]
    q3 = V[:, 2, 3][..., None]
    q = tf.concat([q0, q1, q2, q3], axis=1)
    sel = tf.reshape(tf.to_float(q[:, 0] < 0.0), (-1, 1))
    q = (1.0 - sel) * q - sel * q

    return q


def tf_matrix_from_quaternion(q, eps=1e-10):

    import tensorflow as tf

    # Make unit quaternion
    q_norm = q / (eps + tf.norm(q, axis=1, keep_dims=True))
    q_norm *= tf.constant(2.0 ** 0.5, dtype=tf.float32)
    qq = tf.matmul(
        tf.reshape(q_norm, (-1, 4, 1)),
        tf.reshape(q_norm, (-1, 1, 4))
    )
    M = tf.stack([
        1.0 - qq[:, 2, 2] - qq[:, 3, 3], qq[:, 1, 2] - qq[:, 3, 0],
        qq[:, 1, 3] + qq[:, 2, 0], qq[:, 1, 2] + qq[:, 3, 0],
        1.0 - qq[:, 1, 1] - qq[:, 3, 3], qq[:, 2, 3] - qq[:, 1, 0],
        qq[:, 1, 3] - qq[:, 2, 0], qq[:, 2, 3] + qq[:, 1, 0],
        1.0 - qq[:, 1, 1] - qq[:, 2, 2]
    ], axis=1)

    return M


def tf_skew_symmetric(v):

    import tensorflow as tf

    zero = tf.zeros_like(v[:, 0])

    M = tf.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M


def tf_unskew_symmetric(M):

    import tensorflow as tf

    v = tf.stack([
        0.5 * (M[:, 7] - M[:, 5]),
        0.5 * (M[:, 2] - M[:, 6]),
        0.5 * (M[:, 3] - M[:, 1]),
    ], axis=1)

    return v


# ------------------------------------------------------------
# Architecture related

def bn_act(linout, perform_gcn, perform_bn, activation_fn, is_training,
           data_format):

    import tensorflow as tf

    """ Perform batch normalization and activation """
    if data_format == "NHWC":
        axis = -1
    else:
        axis = 1

    # Global Context normalization on the input
    if perform_gcn:
        # Epsilon to be used in the tf.nn.batch_normalization
        var_eps = 1e-3
        # get mean variance for single sample (channel-wise, note that we omit
        # axis=1 since we are expecting a size of 1 in that dimension)
        mean, variance = tf.nn.moments(linout, axes=[2], keep_dims=True)
        # Use tensorflow's nn.batchnorm
        linout = tf.nn.batch_normalization(
            linout, mean, variance, None, None, var_eps)

    if perform_bn:
        with tf.variable_scope("bn", reuse=tf.AUTO_REUSE):
            linout = tf.layers.batch_normalization(
                inputs=linout,
                center=True, scale=True,
                training=is_training,
                trainable=False,
                axis=axis,
            )

    if activation_fn is None:
        output = linout
    else:
        output = activation_fn(linout)

    return output


def pad_cyclic(tensor, paddings):

    import tensorflow as tf

    ndim = len(paddings)
    for _dim, _pad in zip(xrange(ndim), paddings):

        pad_list = []
        if _pad[0] > 0:
            # Padding to put at front
            slice_st = [slice(None, None)] * ndim
            slice_st[_dim] = slice(-_pad[0], None)
            pad_list += [tensor[tuple(slice_st)]]

        # Original
        pad_list += [tensor]

        if _pad[1] > 0:
            # Padding to put at back
            slice_ed = [slice(None, None)] * ndim
            slice_ed[_dim] = slice(None, _pad[1])
            pad_list += [tensor[tuple(slice_ed)]]

        if len(pad_list) > 1:
            # Concatenate to do padding
            tensor = tf.concat(pad_list, axis=_dim)

    return tensor


def conv1d_pad_cyclic(inputs, ksize, numconv, data_format="NCHW"):
    in_shp = tf_get_shape_as_list(inputs)
    ksize = 2 * (ksize // 2 * numconv) + 1

    if data_format == "NCHW":
        assert (ksize < in_shp[-1]) or (in_shp[-1] == -1)
        if np.mod(ksize, 2) == 0:
            paddings = [
                [0, 0], [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2]
            ]
        else:
            paddings = [
                [0, 0], [0, 0], [0, 0], [ksize // 2, ksize // 2]
            ]
    else:
        assert (ksize < in_shp[-2]) or (in_shp[-2] == -1)
        if np.mod(ksize, 2) == 0:
            paddings = [
                [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2], [0, 0]
            ]
        else:
            paddings = [
                [0, 0], [0, 0], [ksize // 2, ksize // 2], [0, 0]
            ]
    inputs = pad_cyclic(inputs, paddings)

    return inputs


def get_W_b_conv1d(in_channel, out_channel, ksize, dtype=None):

    import tensorflow as tf

    if dtype is None:
        dtype = tf.float32

    fanin = in_channel * ksize
    W = tf.get_variable(
        "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
        # initializer=tf.random_normal_initializer(stddev=0.02),
    )
    b = tf.get_variable(
        "biases", shape=[out_channel], dtype=dtype,
        initializer=tf.zeros_initializer(),
    )

    tf.summary.histogram("W", W)
    tf.summary.histogram("b", b)

    return W, b


def conv1d_layer(inputs, ksize, nchannel, activation_fn, perform_bn,
                 perform_gcn, is_training, perform_kron=False,
                 padding="CYCLIC", data_format="NCHW",
                 act_pos="post"):

    import tensorflow as tf

    assert act_pos == "pre" or act_pos == "post"

    # Pad manually
    if padding == "CYCLIC":
        if ksize > 1:
            inputs = conv1d_pad_cyclic(
                inputs, ksize, 1, data_format=data_format)
        cur_padding = "VALID"
    else:
        cur_padding = padding

    in_shp = tf_get_shape_as_list(inputs)
    if data_format == "NHWC":
        in_channel = in_shp[-1]
        ksizes = [1, 1, ksize, 1]
    else:
        in_channel = in_shp[1]
        ksizes = [1, 1, 1, ksize]

    assert len(in_shp) == 4

    # # Lift with kronecker
    # if not is_first:
    #     inputs = tf.concat([
    #         inputs,
    #         kronecker_layer(inputs),
    #     ], axis=-1)

    self_ksize = ksize
    do_add = False

    # If pre activation
    if act_pos == "pre":
        inputs = bn_act(inputs, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format)

    # Normal convolution
    with tf.variable_scope("self-conv"):
        W, b = get_W_b_conv1d(in_channel, nchannel, self_ksize)
        # tf.summary.histogram("W", W)
        # tf.summary.histogram("b", b)
        # Convolution in the valid region only
        linout = tf.nn.conv2d(
            inputs, W, [1, 1, 1, 1], cur_padding, data_format=data_format)
        linout = tf.nn.bias_add(linout, b, data_format=data_format)

    # If post activation
    output = linout
    if act_pos == "post":
        output = bn_act(linout, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format)

    return output


def conv1d_resnet_block(inputs, ksize, nchannel, activation_fn, is_training,
                        midchannel=None, perform_bn=False, perform_gcn=False,
                        padding="CYCLIC", act_pos="post", data_format="NCHW"):

    import tensorflow as tf

    # In case we want to do a bottleneck layer
    if midchannel is None:
        midchannel = nchannel

    # don't activate conv1 in case of midact
    conv1_act_fn = activation_fn
    if act_pos == "mid":
        conv1_act_fn = None
        act_pos = "pre"

    # Pass branch
    with tf.variable_scope("pass-branch"):
        # passthrough to be used when num_outputs != num_inputs
        in_shp = tf_get_shape_as_list(inputs)
        if data_format == "NHWC":
            in_channel = in_shp[-1]
        else:
            in_channel = in_shp[1]
        if in_channel != nchannel:
            cur_in = inputs
            # Simply change channels through 1x1 conv
            with tf.variable_scope("conv"):
                cur_in = conv1d_layer(
                    inputs=inputs, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
            orig_inputs = cur_in
        else:
            orig_inputs = inputs

    # Conv branch
    with tf.variable_scope("conv-branch"):
        cur_in = inputs
        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("preconv"):
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
                cur_in = activation_fn(cur_in)

        # Main convolution
        with tf.variable_scope("conv1"):
            # right branch
            cur_in = conv1d_layer(
                inputs=cur_in, ksize=ksize,
                nchannel=nchannel,
                activation_fn=conv1_act_fn,
                perform_bn=perform_bn,
                perform_gcn=perform_gcn,
                is_training=is_training,
                padding=padding,
                act_pos=act_pos,
                data_format=data_format,
            )

        # Main convolution
        with tf.variable_scope("conv2"):
            # right branch
            cur_in = conv1d_layer(
                inputs=cur_in, ksize=ksize,
                nchannel=nchannel,
                activation_fn=activation_fn,
                perform_bn=perform_bn,
                perform_gcn=perform_gcn,
                is_training=is_training,
                padding=padding,
                act_pos=act_pos,
                data_format=data_format,
            )

        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("postconv"):
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
                cur_in = activation_fn(cur_in)

    # Crop lb or rb accordingly
    if padding == "VALID" and ksize > 1:
        # Crop pass branch results
        if np.mod(ksize, 2) == 0:
            crop_st = ksize // 2 - 1
        else:
            crop_st = ksize // 2
            crop_ed = ksize // 2
            if data_format == "NHWC":
                orig_inputs = orig_inputs[:, :,  crop_st:-crop_ed, :]
            else:
                orig_inputs = orig_inputs[:, :, :, crop_st:-crop_ed]

    return cur_in + orig_inputs


def linear(input_, outputSize, activation_fn = None, name = 'linear'):

    import tensorflow as tf

    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):

        w = tf.get_variable('w_linear', [shape[1], outputSize], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('bias', [outputSize], initializer=tf.constant_initializer(0.0))

        out = tf.matmul(input_,w) + b

        if activation_fn != None:
            return activation_fn(out)
        else:
            return out

    # W = tf.get_variable(
    #     "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
    #     initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
    #     # initializer=tf.random_normal_initializer(stddev=0.02),
    # )
    # b = tf.get_variable(
    #     "biases", shape=[out_channel], dtype=dtype,
    #     initializer=tf.zeros_initializer(),
    # )

def conv2d(x, outputDim, patchSize, stride, activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):

    with tf.variable_scope(name):

        s = [1, stride[0], stride[1], 1]
        kernelShape = [patchSize, patchSize, x.get_shape().as_list()[-1], outputDim]

        w = tf.get_variable('w', kernelShape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(x, w, s, padding)

        b = tf.get_variable('bias', [outputDim], initializer=tf.constant_initializer(0.0))
        out = activation_fn(conv + b)

    return out, w, b

def resnet_reg(cur_input, nb_channels, patchSize, stride, is_training, data_format="NCHW", padding='SAME',
               activation_fn= tf.nn.relu, name ='resblock-reg'):

    with tf.variable_scope(name):

        output, w1, b1 = conv2d(cur_input, nb_channels, patchSize, stride, padding=padding, name='res1')
        output = bn_act(output, False, True, None, is_training, data_format)

        output, w2, b2 = conv2d(output, nb_channels, patchSize, stride, padding=padding, name='res2')
        output = bn_act(output, False, True, None, is_training, data_format)

        output = tf.add(output, cur_input)

    return output


def regression_layer(cur_input, nb_channels, patch, stride, nb_fc, perform_gcn, perform_bn, activation_fn, is_training,
                     representation, data_format="NCHW"):

    cur_input = bn_act(cur_input, perform_gcn, perform_bn, activation_fn, is_training, data_format)

    cur_input = tf.expand_dims(cur_input, axis=3)
    outputs, w, b = conv2d(cur_input, nb_channels, patch, stride)
    print(outputs.shape)
    print('here ^')
    #
    #outputs = resnet_reg(outputs, nb_channels, patch, [1, 1], is_training)
    #print(outputs.shape)
    # outputs = bn_act(outputs, perform_gcn, perform_bn, activation_fn, is_training, data_format)
    # outputs, w, b = conv2d(outputs, nb_channels*2, patch, [1, 1])

    shape = outputs.get_shape()
    num_features = shape[1:4].num_elements()
    outputs = tf.reshape(outputs, [-1, num_features])
    print(outputs.shape)

    l1 = linear(outputs, nb_fc, activation_fn=tf.nn.relu, name='linear_1')
    # R_hat = linear(l1, 3, name='R_hat')
    # t_hat = linear(l1, 3, name='t_hat')

    if representation == 'lie':
        Rt_hat = linear(l1, 6, name='Rt_hat')
        R_hat = tf.transpose(tf.stack([Rt_hat[:, 0], Rt_hat[:, 1], Rt_hat[:, 2]]))
        t_hat = tf.transpose(tf.stack([Rt_hat[:, 3], Rt_hat[:, 4], Rt_hat[:, 5]]))

    elif representation == 'quat':
        Rt_hat = linear(l1, 7, name='Rt_hat')
        R_hat = tf.transpose(tf.stack([Rt_hat[:, 0], Rt_hat[:, 1], Rt_hat[:, 2], Rt_hat[:, 3]]))
        t_hat = tf.transpose(tf.stack([Rt_hat[:, 4], Rt_hat[:, 5], Rt_hat[:, 6]]))

    elif representation == 'linear':
        Rt_hat = linear(l1, 12, name='Rt_hat')
        R_hat = tf.transpose(tf.stack([Rt_hat[:, 0], Rt_hat[:, 1], Rt_hat[:, 2],
                                       Rt_hat[:, 3], Rt_hat[:, 4], Rt_hat[:, 5],
                                       Rt_hat[:, 6], Rt_hat[:, 7], Rt_hat[:, 8]]))
        t_hat = tf.transpose(tf.stack([Rt_hat[:, 9], Rt_hat[:, 10], Rt_hat[:, 11]]))

    else:
        print('Error in the representation')
        R_hat = []
        t_hat =[]
        exit(10)


    return R_hat, t_hat


def globalmax_pool1d(inputs):

    import tensorflow as tf

    with tf.variable_scope('max_pool'):
        outputs = tf.reduce_max(inputs, axis=2)

    return outputs

def globalmean_pool1d(inputs):

    import tensorflow as tf

    with tf.variable_scope('mean_pool'):
        outputs = tf.reduce_mean(inputs, axis=2)

    return outputs

def tf_matrix_vector_mul(M, v):

    import tensorflow as tf

    # print(v.shape)

    sh = tf.shape(v)
    M = tf.expand_dims(M, 1)
    M_ = tf.tile(M, [1, sh[1], 1, 1])
    m0 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 0, :], v), axis=2), (sh[0], sh[1], 1))
    m1 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 1, :], v), axis=2), (sh[0], sh[1], 1))
    m2 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 2, :], v), axis=2), (sh[0], sh[1], 1))

    T = tf.concat([m0, m1, m2], axis=2)
    # print(T.shape)

    return T

def tf_matrix4_vector_mul(M, v):

    import tensorflow as tf

    # print(v.shape)

    sh = tf.shape(v)
    M = tf.expand_dims(M, 1)
    M_ = tf.tile(M, [1, sh[1], 1, 1])
    m0 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 0, :], v), axis=2), (sh[0], sh[1], 1))
    m1 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 1, :], v), axis=2), (sh[0], sh[1], 1))
    m2 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 2, :], v), axis=2), (sh[0], sh[1], 1))
    m3 = tf.reshape(tf.reduce_sum(tf.multiply(M_[:, :, 3, :], v), axis=2), (sh[0], sh[1], 1))
    T = tf.concat([m0, m1, m2, m3], axis=2)
    # print(T.shape)

    return T

def tf_add_vectors(v, u):

    import tensorflow as tf

    sh = tf.shape(v)
    u = tf.expand_dims(u, 1)
    u_ = tf.tile(u, [1, sh[1], 1])

    y = tf.add(v, u_)

    return y


def tf_mul_vectors(u, v):

    import tensorflow as tf

    # y = tf.einsum('abi,abj->abij', u, v)
    y = tf.matmul(u, v, transpose_a = True)

    return y


def geman_mcclure(x, alpha = 10.):

    import tensorflow as tf

    y = tf.norm(x, axis=2)


    sh = tf.shape(x)
    alpha = tf.constant(alpha, shape=[1, 1])
    alpha = tf.tile(alpha, [sh[0], sh[1]])
    l = tf.square(y)/2
    l = l/(tf.square(alpha) + tf.square(y))

    return l

def l1(x):
    import tensorflow as tf

    return tf.abs(tf.norm(x, axis=2))

def l2(x):
    import tensorflow as tf

    return tf.reduce_sum(tf.square(x), axis=2)

def l05(x):

    import tensorflow as tf

    return 2*tf.sqrt(tf.abs(tf.norm(x, axis=2)))

def np_matrix4_vector_mul(M, v):

    import numpy as np

    # print(v.shape)

    sh = v.shape
    M = np.expand_dims(M, 0)
    M_ = np.tile(M, [sh[0], 1, 1])
    m0 = np.reshape(np.sum(np.multiply(M_[:, 0, :], v), axis=1), (sh[0], 1))
    m1 = np.reshape(np.sum(np.multiply(M_[:, 1, :], v), axis=1), (sh[0], 1))
    m2 = np.reshape(np.sum(np.multiply(M_[:, 2, :], v), axis=1), (sh[0], 1))
    m3 = np.reshape(np.sum(np.multiply(M_[:, 3, :], v), axis=1), (sh[0], 1))

    T = np.concatenate([m0, m1, m2, m3], axis=1)
    # print(T.shape)

    return T

#
# ops.py ends here
