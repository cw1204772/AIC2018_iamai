# 2018 NVIDIA AI City Challenge

Hi! We are participating team 37, **"iamai"**,  of 2018 NVIDIA AI City Challenge Track 3.  
This is also the implementation of Chih-Wei Wu, Chih-Ting Liu, Chen-En Jiang, Wei-Chih Tu, Shao-Yi Chien **"Vehicle Re-Identification with the Space-Time Prior"** _CVPRW, 2018_.  
It is an end-to-end vehicle detection, tracking, re-identification system.

To clone this repo, please execute:
```
git clone --recurse-submodules https://github.com/cw1204772/AIC2018_iamai.git  
```

If you've already clone this repo but haven't clone the submodule (`Tracking/iou-tracker`), execute:
```
git submodule init
git submodule update
```

If you experience any bugs or problems, please contact us. (cwwu@media.ee.ntu.edu.tw)

## Requirements

It requires both python 2 and 3 to run our system.
* Python 2.7 or newer:  
  Please install [detectron](https://github.com/facebookresearch/Detectron), a powerful open-sourced object detector thanks to Facebook. Please refer to the [INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) to install all dependencies for inference.
* Python 3.5 or newer:  
  Run `pip3 install -r requirements.txt` to install all dependence packages.

## Demo

Hurray!  
We've managed to create a script for running the entire system!  
First, Download all 2018 NVIDIA AI City Challenge Track 3 videos into `<DATASET_DIR>`.  
Then, you need to download the pre-trained Re-ID CNN model. Please email your full name and affiliation to the contact person (cwwu@media.ee.ntu.edu.tw) for obtaining the download link. Download the model to `ReID/ReID_CNN/`.  
Last, execute:
```
./run.sh <DATASET_DIR> <WORK_DIR>
```
__\*\*Important**__  
`<WORK_DIR>` will be the storage place for intermediate product of our system. Make sure there is enough space for `<WORK_DIR>`! (We estimate at least 1.2TB of space!:open_mouth: Because we will unpact video into images for detection.)  
Also, please use absolute path for both `<DATASET_DIR>` and `<WORK_DIR>`.

Expect to wait for a few days, or maybe, weeks, depending on your machine. (Yes, we are not exaggerating. Detection itself took weeks on our machine with 1 GTX1080Ti)  
The final result will show up here: `<WORK_DIR>/MCT/fasta/track3.txt`.  
(Assuming there are no bugs!:smiley:)

## Detail Guide

Here, we provide detail instructions for each stage of our system.

### Detection

We use [detectron](https://github.com/facebookresearch/Detectron) for detection. Please refer to the [INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) to install caffe2 and other dependencies for inference.

1. Convert all the videos to frames
   
   We assume that you organize your videos dataset as the directory structure below:
   ```
   /path/to/AIC_videos_dataset
     |__Loc1_1.mp4
     |__...
     |__Loc4_3.mp4
   ```
   
   After running:
   ```
   python2 Utils/convert.py --video_dir_path /path/to/AIC_videos_dataset --images_dir_path /path/to/AIC_images_dataset
   ```
   
   And the new directory structure will become:
   ```
   /path/to/AIC_frames_dataset
     |__Loc1_1
     |  |__<frame_1>.jpg
     |  |__...
     |  |__<frame_N>.jpg
     |__...
     |__Loc4_3
   ```

2. Infer frames for every locations
   ```
   cd $AIC2018_iamai/Detection/
   python2 tools/infer_simple_txt.py \
       --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
       --output-dir /path/to/submit \
       --image-ext jpg \
       --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
       /path/to/AIC_dataset
   ```
3. Suppress non-realistic bounding boxes
   ```
   cd $AIC2018_iamai/Detection/
   python2 tools/suppress.py --in_txt_file_path <input_txt> --out_txt_file_path <output_txt> --threshold 1e-5 --upper_area 1e5 --lower_side 25 --aspect_ratio 5
   ```

### Tracking

We use our optimized version of [iou-tracker](https://github.com/bochinski/iou-tracker) for tracking.  
It will link detections into tracklets by simple IOU constraint within a video.
To use it, try:

```
python3 Tracking/iou-tracker/demo.py [-h] -d DETECTION_PATH -o OUTPUT_PATH [-sl SIGMA_L]
                                     [-sh SIGMA_H] [-si SIGMA_IOU] [-tm T_MIN]
```

### Post-Tracking

In this step, we will extract keypoint images within each tracklet.  
To use it, try:

```
python3 ReID/Post_tracking.py [-h] [--dist_th DIST_TH] [--size_th SIZE_TH]
                              [--mask MASK] [--img_dir IMG_DIR]
                              tracking_csv video output
```

### Train CNN Feature Extractor

If you wish to train a CNN feature extractor from scratch, please refer to `ReID/ReID_CNN/Readme.md`.  
Or, you can download our pre-trained model. Please email your full name and affiliation to the contact person (cwwu@media.ee.ntu.edu.tw) for obtaining the download link.

### Single Camera Tracking

In this step, we associate tracklets within a video by comparing features and space-time information.  
To use it, try:

```
python3 ReID/SCT.py [-h] [--window WINDOW] [--f_th F_TH] [--b_th B_TH] [--verbose]
                    --reid_model REID_MODEL --n_layers N_LAYERS
                    [--batch_size BATCH_SIZE]
                    pkl output
```

### Multi Camera Matching

There are a few matching methods to choose from, including the most successful `re-rank-4`.
To use it, try:

```
python3 ReID/MCT.py [-h] [--dump_dir DUMP_DIR] [--method METHOD] [--cluster CLUSTER]
                    [--normalize] [--k K] [--n N] [--sum SUM] [--filter FILTER]
                    tracks_dir output_dir
```

## Tools

Here is a visualization tool we create to cheer your eyes during the tedious running process.

```
python3 Utils/visualize.py [-h] [--w W] [--h H] [--fps FPS] [--length LENGTH]
                           [--delimiter DELIMITER] [--offset OFFSET]
                           [--frame_pos FRAME_POS] [--bbox_pos BBOX_POS]
                           [--id_pos ID_POS] [--score_pos SCORE_POS]
                           [--score_th SCORE_TH] [--cam CAM] [--cam_pos CAM_POS]
                           [--ss SS] [--wh_mode]
                           INPUT_VIDEO OUTPUT_VIDEO LABEL_FILE MODE
```


## References

* NVIDIA AI City Challenge. https://www.aicitychallenge.org/, 2018.
* R. Girshick, I. Radosavovic, G. Gkioxari, P. Dollar, and K. He. Detectron. https://github.com/facebookresearch/detectron, 2018.
* E. Bochinski, V. Eiselein, and T. Sikora. High-speed tracking-by-detection without using image information. https://github.com/bochinski/iou-tracker. AVSS, 2017.

## Citing

```
@inproceedings{wu2018vreid,
  title={Vehicle Re-Identification with the Space-Time Prior},
  author={Wu, Chih-Wei and Liu, Chih-Ting and Jiang, Chen-En and Tu, Wei-Chih and Chien, Shao-Yi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshop},
  year={2018},
}
```
