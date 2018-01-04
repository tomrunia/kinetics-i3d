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
import cv2

from cortex.vision.video_reader import VideoReaderOpenCV

################################################################################

video_dir_in  = "/home/tomrunia/data/Kinetics/Full/dump-from-amir/videos/"
image_dir_out = "/home/tomrunia/data/Kinetics/Full/videos_as_images/"

# Resize smaller side to this dimension (px)
resize_small_side = 256

################################################################################

# Get all action categories (subdirectories)
categories_dirs = [x[0] for x in os.walk(video_dir_in)]
categories_dirs = categories_dirs[1:] # skip current dir
categories_dirs.sort()

for cat_id, categories_dir in enumerate(categories_dirs):

    category = os.path.basename(categories_dir)
    category_im_dir = os.path.join(image_dir_out, category)

    if not os.path.exists(category_im_dir):
        os.makedirs(category_im_dir)

    # Get all videos in current action class (video files)
    video_files = glob.glob(os.path.join(categories_dir, "*.mp4"))
    video_files.sort()

    for vid_id, video_file in enumerate(video_files):

        video_filename, _ = os.path.splitext(os.path.basename(video_file))
        video_im_dir = os.path.join(category_im_dir, video_filename)

        if not os.path.exists(video_im_dir):
            os.makedirs(video_im_dir)

        # Open the video and set resizing option
        video = VideoReaderOpenCV(video_file, as_float=False, resize_small_side=resize_small_side)

        # Do we already have all the frames?
        images = glob.glob(os.path.join(video_im_dir, "*.jpg"))
        if len(images) == video.length:
            print("Skipping because already processed: {}".format(video_file))
            continue

        # Read all the frames into memory (resized)
        frames = video.all_frames()

        print("[{}/{} - {}] Extracting frames for video {}/{}: {}".format(
            cat_id+1, len(categories_dirs), category, vid_id+1, len(video_files), video_file))

        for frame_idx in range(len(frames)):
            image_file = os.path.join(video_im_dir, "{:06d}.jpg".format(frame_idx))
            cv2.imwrite(image_file, frames[frame_idx])