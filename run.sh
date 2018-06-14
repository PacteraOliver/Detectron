#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter4999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter5999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter6999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter7999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter8999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter9999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter10999.pkl data/val

python2 tools/infer_simple_train.py \
 --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x_custom.yaml \
 --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg \
 --wts weights/model_step/model_iter11999.pkl data/val