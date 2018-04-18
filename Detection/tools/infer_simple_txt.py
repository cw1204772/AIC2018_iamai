#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    for root_dir_path_1, sub_dir_path_list_1, sub_file_path_list_1 in os.walk(args.im_or_folder):
        sub_dir_path_list_1 = sorted(sub_dir_path_list_1)
        for i, sub_dir_path_1 in enumerate(sub_dir_path_list_1):
            for root_dir_path_2, sub_dir_path_list_2, sub_file_path_list_2 in os.walk(os.path.join(root_dir_path_1, sub_dir_path_1)):
                sub_file_path_list_2 = sorted(sub_file_path_list_2)
                out_file = open(os.path.join(args.output_dir, sub_dir_path_1 + "_Det_ffasta.txt"), "wb")
                for img_idx, sub_file_path_2 in enumerate(sub_file_path_list_2):
                    im = cv2.imread(os.path.join(root_dir_path_2, sub_file_path_2))
                    timers = defaultdict(Timer)
                    t = time.time()
                    if (img_idx + 1) % 1000 == 0:
                        sys.stdout.write("\rFinish {} images\n".format(img_idx + 1))
                        sys.stdout.flush()
                    with c2_utils.NamedCudaScope(0):
                        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                            model, im, None, timers=timers
                        )
                        if isinstance(cls_boxes, list):
                            cls_boxes, cls_segms, cls_keyps, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
                        if cls_boxes is None or cls_boxes.shape[0] == 0:
                            continue
                        obj_idx = 0
                        for cls_box, cls in zip(cls_boxes, classes):
                            if int(cls) != 3 and int(cls) != 6:
                                continue
                            out_file.write("{},{},{},{},{},{},{}\n".format(img_idx + 1, obj_idx + 1, cls_box[0], cls_box[1], cls_box[2] - cls_box[0], cls_box[3] - cls_box[1], cls_box[4]))
                            obj_idx += 1
                out_file.close()
            print("Finish {} / {} of video sequences".format(i + 1, len(sub_dir_path_list_1)))
        break

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
