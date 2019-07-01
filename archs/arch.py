# arch.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import tensorflow as tf

from ops import conv1d_layer, conv1d_resnet_block, globalmax_pool1d, regression_layer


def build_graph(x_in, is_training, config):

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    print(cur_input.shape)
    idx_layer = 0
    numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block


    # First convolution
    with tf.variable_scope("hidden-input"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
        )
        print(cur_input.shape)

        concat = globalmax_pool1d(cur_input)

    for _ksize, _nchannel in zip(
            [ksize] * numlayer, [nchannel] * numlayer):
        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            cur_input = conv1d_block(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )

            print(cur_input.shape)

        idx_layer += 1

        concat = tf.concat([concat, globalmax_pool1d(cur_input)], axis=1)

    if config.net_concat_post:
        concat, _ = tf.nn.top_k(tf.transpose(cur_input, perm=[0, 1, 3, 2]), k=13)
        concat = tf.squeeze(concat, axis=1)
        concat = tf.transpose(concat, perm=[0, 2, 1])

    print(concat.shape)

    with tf.variable_scope("regression"):
    # with tf.variable_scope("output"):
        R_hat, t_hat = regression_layer(concat,
                                 nb_channels=8,
                                 patch=3,
                                 stride=[1, 2],
                                 nb_fc=256,
                                 activation_fn=None,
                                 perform_bn=False,
                                 perform_gcn=True,
                                 is_training=is_training,
                                 data_format="NHWC",
                                 representation= config.representation)


    with tf.variable_scope("output"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        #  Flatten
        cur_input = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))

    logits = cur_input
    print(cur_input.shape)


    return logits, R_hat, t_hat




#
# arch.py ends here
