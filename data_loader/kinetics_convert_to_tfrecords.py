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
import glob

import numpy as np
import tensorflow as tf
import cv2

from cortex.vision.video_reader import VideoReaderOpenCV


################################################################################

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_video_to_tfrecords(video_dir, example, resize_small_side=None):

    print("Converting video to TFRecords: {}".format(example['video_id']))

    # Build the full video file path
    video_path = os.path.join(
        video_dir, example['class_label'], "{}_{:06d}_{:06d}.mp4".format(
            example['video_id'], example['frame_start'], example['frame_end']))

    # Open the video and set resizing option
    video = VideoReaderOpenCV(
        video_path, as_float=False, resize_small_side=resize_small_side)

    # Read all the frames into memory (resized)
    frames = video.all_frames()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'video/num_frames':  _int64_feature(frames.shape[0]),
        'video/height':      _int64_feature(frames.shape[1]),
        'video/width':       _int64_feature(frames.shape[2]),
        'video/channels':    _int64_feature(frames.shape[3]),
        'video/filename':    _bytes_feature(tf.compat.as_bytes(example['video_id'])),
        'video/class_label': _int64_feature(example['class_id']),
        'video/class_text':  _bytes_feature(tf.compat.as_bytes(example['class_label'])),
        # Note the .tobytes() in the line below
        'video/frames':      _bytes_feature(tf.compat.as_bytes(frames.tobytes()))}))

    return tf_example

def write_label_mapping(train_split_file, label_map_file):
    class_index = 1
    classes = []
    f_train_split = open(split_list_file, 'r')
    f_label_map   = open(label_map_file, 'w')
    f_train_split.readline()  # skip file header
    for line in f_train_split:
        parts = line.rstrip().split(',')
        class_label = parts[0].replace('"', '')
        if class_label not in classes:
            f_label_map.write("{},{}\n".format(class_index, class_label))
            classes.append(class_label)
            class_index += 1
    f_train_split.close()
    f_label_map.close()

def read_label_mapping(label_mapping_file):
    assert os.path.exists(label_mapping_file)
    label_mapping = {}
    with open(label_mapping_file) as f:
        for line in f:
            parts = line.rstrip().split(',')
            label_mapping[parts[1]] = int(parts[0])
    return label_mapping

def collect_all_videos(video_path):
    examples = []
    # Get all action categories (subdirectories)
    categories_dirs = [x[0] for x in os.walk(video_path)]
    categories_dirs = categories_dirs[1:] # skip current dir
    categories_dirs.sort()
    for cat_id, categories_dir in enumerate(categories_dirs):
        category = os.path.basename(categories_dir)
        # Get all videos in current action class (video files)
        video_files = glob.glob(os.path.join(categories_dir, "*.mp4"))
        video_files.sort()
        for vid_id, video_file in enumerate(video_files):
            video = {'category': category, 'video_file': video_file}
            examples.append(video)
    return examples

def read_example_list(split_list_file, label_mapping):
    print("Reading example from: {}".format(split_list_file))
    assert os.path.exists(split_list_file)
    examples = []
    with open(split_list_file, 'r') as f:
        f.readline()  # skip file header
        for line in f:
            parts = line.rstrip().split(',')
            class_label = parts[0].rstrip().replace('"', '')
            assert class_label in label_mapping.keys()
            example = {
                'video_id': parts[1],
                'frame_start': int(parts[2]),
                'frame_end': int(parts[3]),
                'class_id': label_mapping[class_label],
                'class_label': class_label,
            }
            examples.append(example)
    print("Found {} examples.".format(len(examples)))
    return examples


def convert_kinetics_to_tfrecords(video_path, split_list_file, label_mapping_file,
                                  output_path, max_dim_resize=256,
                                  examples_per_file=20):

    # Read label mapping from file
    label_mapping = read_label_mapping(label_name_file)

    # Read 'train', 'val' or 'test' split from CSV file
    example_list = read_example_list(split_list_file, label_mapping)

    num_tfrecord_files = int(np.ceil(len(example_list)/examples_per_file))
    example_idx_offset = 0

    # Set tfrecords configuration, i.e. compression
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    for tf_record_idx in range(num_tfrecord_files):

        # Open a new TFRecords file
        tfrecords_file_name = os.path.join(output_path, "{:06d}.tfrecords".format(tf_record_idx))
        with tf.python_io.TFRecordWriter(tfrecords_file_name, options) as writer:

            for i in range(example_idx_offset, min(example_idx_offset+examples_per_file, len(example_list))):
                # Convert video to tfrecord and write it to tfrecords
                example = convert_video_to_tfrecords(video_path, example_list[i], resize_small_side)
                writer.write(example.SerializeToString())

        # Prepare for next batch in tfrecords file
        example_idx_offset += examples_per_file


if __name__ == "__main__":

    kinetics_path = "/home/tomrunia/data/Kinetics/Full/"

    video_path = os.path.join(kinetics_path, "videos")
    label_name_file = os.path.join(kinetics_path, "label_mapping.txt")
    split_list_file = os.path.join(kinetics_path, "splits/kinetics_train.csv")
    tfrecords_path  = os.path.join(kinetics_path, "tfrecords")

    # First step: build the label mapping
    #write_label_mapping(split_list_file, label_name_file)

    # Resize smaller side to this dimension (px)
    resize_small_side = 256

    # Number of examples per TFRecords file
    examples_per_file = 25

    # Size computation of Kinetics dataset
    # Number of videos per tfrecords file
    # ~300 images per video, ~120 kilobytes per image (uncompressed) = 36Mb per video (!)
    # Kinetics has 300k videos => 11 Terabytes (!!)

    # Even with tfrecord's GZIP compression this is 1.6 Terabytes !!
    convert_kinetics_to_tfrecords(
        video_path, split_list_file, label_name_file,
        tfrecords_path, resize_small_side, examples_per_file
    )


