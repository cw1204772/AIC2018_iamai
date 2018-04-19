#!/bin/bash
set -e

DATASET_DIR=$1
WORK_DIR=$2

SRC_DIR=.
NAME=fasta
REID_MODEL=$SRC_DIR/ReID/ReID_CNN/model_880_base.ckpt

DATASET_IMG_DIR=$WORK_DIR/img
DETECTION_DIR=$WORK_DIR/detections/$NAME
TRACKING_DIR=$WORK_DIR/tracking/$NAME
POST_TRACKING_DIR=$WORK_DIR/post_tracking/$NAME
SCT_DIR=$WORK_DIR/SCT/$NAME
MCT_DIR=$WORK_DIR/MCT/$NAME
VIDEO_DIR=$DATASET_DIR
META_DIR=$SRC_DIR/Dataset_info

mkdir -p $MCT_DIR

LOC=(1 2 3 4)
N_SEQ=(4 6 2 3)

# Detection
echo "[Detection]"
# 1. Convert video data into img data
mkdir -p $DATASET_IMG_DIR
python2 Utils/convert.py --videos_dir_path $DATASET_DIR --images_dir_path $DATASET_IMG_DIR
# 2. Run detector
mkdir -p $DETECTION_DIR
cd $SRC_DIR/Detection
python2 tools/infer_simple_txt.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir $DETECTION_DIR \
    --image-ext jpg \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    $DATASET_IMG_DIR
cd ..
# 3. Filter out impractical detections
for i in $(seq 0 3); do
  for s in $(seq 1 ${N_SEQ[i]}); do
    SEQ_NAME=Loc${LOC[i]}_${s}
    python2 $SRC_DIR/Detection/tools/suppress.py --in_txt_file_path $DETECTION_DIR/${SEQ_NAME}_Det_ffasta.txt \
                                                 --out_txt_file_path $DETECTION_DIR/${SEQ_NAME}.txt \
                                                 --threshold 1e-5 --upper_area 1e5 --lower_side 25 --aspect_ratio 5
  done
done

# Tracking
mkdir -p $TRACKING_DIR
for i in $(seq 0 3); do
  for s in $(seq 1 ${N_SEQ[i]}); do
    SEQ_NAME=Loc${LOC[i]}_${s}
    echo $SEQ_NAME

    echo "[Tracking]"
    if [ "$i" == 0 ] || [ "$i" == 1 ]; then
      python3 $SRC_DIR/Tracking/iou-tracker/demo.py -d $DETECTION_DIR/${SEQ_NAME}.txt \
                                                    -o $TRACKING_DIR/${SEQ_NAME}.csv \
                                                    -sl 0.2 -sh 0.8 -si 0.5 -tm 30
    else
      python3 $SRC_DIR/Tracking/iou-tracker/demo.py -d $DETECTION_DIR/${SEQ_NAME}.txt \
                                                    -o $TRACKING_DIR/${SEQ_NAME}.csv \
                                                    -sl 0.2 -sh 0.8 -si 0.7 -tm 15
    fi
  done
done

# Post tracking
mkdir -p $POST_TRACKING_DIR
for i in $(seq 0 3); do
  for s in $(seq 1 ${N_SEQ[i]}); do
    SEQ_NAME=Loc${LOC[i]}_${s}
    echo $SEQ_NAME

    echo "[Post Tracking]"
    python3 $SRC_DIR/ReID/Post_tracking.py $TRACKING_DIR/${SEQ_NAME}.csv \
                                 $VIDEO_DIR/${SEQ_NAME}.mp4 \
                                 $POST_TRACKING_DIR \
                                 --img_dir ${POST_TRACKING_DIR}/${SEQ_NAME}
  done
done

# SCT
mkdir -p $SCT_DIR
for i in $(seq 0 3); do
  for s in $(seq 1 ${N_SEQ[i]}); do
    SEQ_NAME=Loc${LOC[i]}_${s}
    echo $SEQ_NAME

    echo "[SCT]"
    python3 $SRC_DIR/ReID/SCT.py $POST_TRACKING_DIR/${SEQ_NAME}.pkl \
                                 $SCT_DIR \
                                 --reid_model $REID_MODEL --n_layers 50 \
                                 --window 15 --f_th 100 --b_th 150 \
                                 --batch_size 32
  done
done

# MCT
mkdir -p $MCT_DIR
python3 $SRC_DIR/ReID/MCT.py $SCT_DIR $MCT_DIR --dump_dir $MCT_DIR/dump_final --method re-rank-4 --cluster minibatchkmeans --k 2500 --sum avg --n 20

