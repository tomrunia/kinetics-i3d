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
import sys
import h5py
from io import BytesIO

from PIL import Image

import numpy as np
import tensorflow as tf
import cv2

from cortex.vision.video_reader import VideoReaderOpenCV


################################################################################

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


def video_batch_to_hdf5(video_dir, examples, hdf5_file, resize_small_side=256):

    frame_offsets = [0]
    frames_jpg_bytes = []

    for video_idx in range(len(examples)):

        # Build the full video file path
        example = examples[video_idx]
        video_path = os.path.join(video_dir, example['class_label'], "{}_{:06d}_{:06d}.mp4".format(
                example['video_id'], example['frame_start'], example['frame_end']))

        # Open the video and set resizing option
        video = VideoReaderOpenCV(video_path, False, False, resize_small_side)

        # Read all the frames into memory (resized)
        frames = video.all_frames()

        for frame_idx in range(len(frames)):

            # JPG compression
            ret, buf = cv2.imencode(".jpg", frames[frame_idx])
            frames_jpg_bytes.append(buf.tostring())
            print(video_idx, frame_idx)

        frame_offsets.append(frame_offsets[-1]+len(frames))

    with h5py.File(hdf5_file, 'w') as hf:

        print(frame_offsets)
        print(frames_jpg_bytes)
        hf.create_dataset("jpg_frames", dtype=bytes, data=frames_jpg_bytes)


def convert_kinetics_to_HDF5(video_path, split_list_file, label_mapping_file,
                             output_path, max_dim_resize=256,
                             examples_per_file=20):

    # Read label mapping from file
    label_mapping = read_label_mapping(label_name_file)

    # Read 'train', 'val' or 'test' split from CSV file
    example_list = read_example_list(split_list_file, label_mapping)

    num_files = int(np.ceil(len(example_list)/examples_per_file))
    example_idx_offset = 0

    video_batch_to_hdf5(video_path, example_list[0:3], "/tmp/test.h5", max_dim_resize)
    return

    # for file_idx in range(num_files):
    #
    #     for i in range(example_idx_offset, min(example_idx_offset+examples_per_file, len(example_list))):
    #         # Convert video to tfrecord and write it to tfrecords
    #         #example = convert_video_to_tfrecords(video_path, example_list[i], resize_small_side)
    #         #writer.write(example.SerializeToString())
    #
    #     # Prepare for next batch in tfrecords file
    #     example_idx_offset += examples_per_file


if __name__ == "__main__":

    kinetics_path = "/home/tomrunia/data/Kinetics/Full/"

    video_path = os.path.join(kinetics_path, "videos")
    label_name_file = os.path.join(kinetics_path, "label_mapping.txt")
    split_list_file = os.path.join(kinetics_path, "splits/kinetics_train.csv")
    hdf5_path = os.path.join(kinetics_path, "hdf5")

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
    convert_kinetics_to_HDF5(
        video_path, split_list_file, label_name_file,
        hdf5_path, resize_small_side, examples_per_file
    )


