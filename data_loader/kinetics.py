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


class KineticsDataset(object):

    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir

    def _read_labels(self):
        pass

    def read_video(self):

        video_path = os.path.join(self._dataset_dir, "videos/abseiling/4lTtJeXACp4_000121_000131.mp4")








if __name__ == "__main__":

    dataset_path = "/home/tomrunia/data/Kinetics/Full/dump-from-amir"
    dataset = KineticsDataset(dataset_path)

    dataset.read_video()








