#!/usr/bin/env bash

sudo docker cp . xxxxxxxxxxx:/detectron/

sudo docker cp model_final.pkl xxxxxxxxxxxxxxx:/detectron/weights/

cd weights
wget https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl
cd ..

python2 tools/infer_simple_bbox.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg --wts weights/model_final.pkl data/train

python2 tools/infer_simple_train.py --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x.yaml --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg --wts weights/model_train_60000.pkl data/val

python2 tools/infer_simple_train.py --cfg configs/12_2017_baselines/wider_e2e_faster_rcnn_R-101-FPN_1x.yaml --output-dir /tmp/detectron-visualizations_bbox --image-ext jpg --wts weights/model_fastrcnn_rpn_anchor_16.pkl data/val