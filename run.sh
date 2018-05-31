#!/usr/bin/env bash

cd weights
wget https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl
cd ..

python2 tools/infer_simple.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg --wts weights/model_final.pkl demo