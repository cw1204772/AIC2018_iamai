# NVIDIA AI Challenge

Team name: iamai

Use `git clone --recurse-submodules https://github.com/cw1204772/AIC2018_iamai.git` to clone the repository.  
If you have already git clone the repository but have not clone the submodule, execute:
```
git submodule init
git submodule update
```

## Detection

## Tracking

We use [iou-tracker](https://github.com/bochinski/iou-tracker) for tracking. To use it:

```
python3 Tracking/iou-tracker/demo.py -d <detection_csv> -o <output_csv> -sl 0.1 -sh 0.7 -si 0.5 -tm 2
```

## ReID

**Use in Single Camera Tracking**

```
python3 ReID/SCT.py <tracking_csv> <video> <output_csv> --window 15 --th 150 --reid_model <reid_model> --n_layers <n_layers>
```

**Train on VeRi Dataset**

`cd ReID/ReID_CNN`

1. Create train, query, gallery info .txt files
   ```
   bash setup.sh <dir_to_VeRi>
   ```

2. Training
   * Train classification model with VeRi or VeRi\_ict dataset

     ```
     python3 train.py --info VeRi_train_info.txt --lr 0.001 --batch_size 64 --n_epochs 20 --n_layer 18 --dataset VeRi
     ```

   * Train triplet model with VeRi dataset
     ```
     python3 train.py --info VeRi_train_info.txt --triplet VeRi_triplet.txt --lr 0.001 --batch_size 64 --n_epochs 20 --save_model_dir ./ckpt --n_layer 18 --margin 2
     ``` 

3. dump distance matrix
``` 
python3 compute_VeRi_dis.py [--load_ckpt <model_path>][--n_layer <ResNet_layer>][--gallery_txt <path_to_txt>]
                            [--query_txt <path_to_txt>][--dis_mat <path_and_filename_to_store_mat_file>]
```
4. compute cmc curve
```  
  1. open matlab in the "VeRi_cmc/" directory
  2. open "baseline_evaluation_FACT_776.m" file
  3. change "dis_CNN" mat path, "gt_index",  "jk_index" txt file path
  4. run and get plot
```

## Visualize

Use it to draw csv file onto video.

```
python3 visualize.py [-h] [--w W] [--h H] [--fps FPS] [--length LENGTH]
                     [--delimiter DELIMITER] [--offset OFFSET]
                     [--frame_pos FRAME_POS] [--bbox_pos BBOX_POS]
                     [--id_pos ID_POS] [--score_pos SCORE_POS]
                     [--score_th SCORE_TH] [--cam CAM] [--cam_pos CAM_POS]
                     [--ss SS] [--wh_mode]
                     INPUT_VIDEO OUTPUT_VIDEO LABEL_FILE MODE
```
