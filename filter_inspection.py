# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-01-03


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import cv2

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def save_filters(unused_argv):

    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained

    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            flow_sample = np.load(_SAMPLE_PATHS['flow'])
            tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
            feed_dict[flow_input] = flow_sample

        # Network inspection
        print("Variable Listing:")
        for var in tf.global_variables("RGB/inception_i3d/"):
            if not var.name.startswith('RGB'): continue
            if 'conv_3d' not in var.name: continue
            print("  {} with shape: {}".format(var.name, var.shape))

        # Inspect a conv block in the network
        variable_name = "RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0"
        filter_tensor = [v for v in tf.global_variables() if v.name == variable_name][0]

        # Tensor of shape (T,H,W,CH_IN,CH_OUT)
        filters = sess.run(filter_tensor)
        np.save('./data/conv3d_1a_7x7.npy', filters)

def write_filters_as_gif():

    import imageio

    filters = np.load('./data/conv3d_1a_7x7.npy')

    for filter_idx in range(filters.shape[-1]):

        images = []
        for t in range(7):

            print("Showing filter {} at timestep {}".format(filter_idx+1,t+1))
            filter_map = filters[t,:,:,:,filter_idx]
            filter_map_norm = np.zeros_like(filter_map)
            cv2.normalize(filter_map, filter_map_norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

            filter_map_norm = cv2.resize(filter_map_norm, dsize=(50,50))
            images.append(filter_map_norm)

            # cv2.imshow("Filter", filter_map_norm)
            # key = cv2.waitKey(100)
            # if key == ord('n'):
            #     print("Moving to next filter.")
            #     break

        # Write the GIF to disk
        imageio.mimsave(os.path.join('./data/filter_videos/{:03d}_conv3d_1a_7x7.gif'.format(filter_idx)), images)

def show_time_space_flattened():

    filters = np.load('./data/conv3d_1a_7x7.npy')

    for filter_idx in range(filters.shape[-1]):

        for x in range(7):

            filter_slice = filters[:,:,x,:,filter_idx]
            filter_slice_norm = np.zeros_like(filter_slice)
            cv2.normalize(filter_slice, filter_slice_norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

            filter_slice_norm = cv2.resize(filter_slice_norm, dsize=(100,100))
            filter_slice_norm = np.transpose(filter_slice_norm, axes=(1,0,2))

            if x == 3:
                im_to_write = (filter_slice_norm*255.0).astype(np.uint8)
                cv2.imwrite(os.path.join('./data/filter_yt_slices/{:03d}_conv3d_1a_7x7_x{}.png'.format(filter_idx, 3)), im_to_write)

            cv2.imshow("filter", filter_slice_norm)
            cv2.waitKey(5)

if __name__ == '__main__':

    #save_filters()
    #visualize_filters()
    show_time_space_flattened()
