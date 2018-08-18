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
import re

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.wider_datasets as wider_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

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
    parser.add_argument(
        '--test',
        dest='test',
        help='T or V',
        default='T',
        type=str
    )
    parser.add_argument(
        '--model-name',
        dest='model_name',
        help='Model Name',
        default='Model_final',
        type=str
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

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_wider_dataset = wider_datasets.get_wider_dataset()

    INFER_BOX_ALPHA = 0.3
    INFER_THRESH = 0.3
    INFER_KP_THRESH = 2
    if "model_iter" in args.weights:
        # MODEL_ITER = str(re.match(r"(.*)model_iter(.*)\.pkl", args.weights).group(2))
        MODEL_ITER = str(re.match(r"(.*)model_iter(.*)\.pkl", args.weights).group(2))
    else:
        MODEL_ITER = "260000"

    logger.info("Model Iter: {}".format(MODEL_ITER))

    if args.test == "T":
        submit_mode = "test"
    elif args.test == "V":
        submit_mode = "val"
    elif args.test == "Tr":
        submit_mode = "train"
    elif args.test == "TN":
        submit_mode = "test_new"
    elif args.test == "OUT":
        submit_mode = "clip_out"
    elif args.test == "DEV":
        submit_mode = "val_dev"
    else:
        submit_mode = "default"

    submit_result = []
    result_file_name = 'detectron_class_{}_result_{}_{}_' \
                       'NMS_{}_SOFT_NMS_{}_RPN_NMS_THRESH_{}_PRE_NMS_{}_' \
                       'POST_NMS_{}_BBOX_AUG_{}_' \
                       'Thresh_{}_BoxNumber.txt'.format(
        submit_mode,
        args.model_name,
        MODEL_ITER,
        cfg.TEST.NMS,
        cfg.TEST.SOFT_NMS.ENABLED,
        cfg.TEST.RPN_NMS_THRESH,
        cfg.TEST.RPN_PRE_NMS_TOP_N,
        cfg.TEST.RPN_POST_NMS_TOP_N,
        cfg.TEST.BBOX_AUG.ENABLED,
        INFER_THRESH)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )

        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        result = vis_utils.vis_one_image_bbox_classes(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_wider_dataset,
            box_alpha=INFER_BOX_ALPHA,
            show_class=False,
            thresh=INFER_THRESH,
            kp_thresh=INFER_KP_THRESH
        )
        if result:
            submit_result.extend(result)
        logger.info('Image {}.'.format(i))

    # Write file
    with open(result_file_name, 'wb') as result_file:
        for item in submit_result:
            result_file.write("%s\n" % item)

    logger.info(
        'The result file has been written in {}'.format(result_file_name)
    )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
