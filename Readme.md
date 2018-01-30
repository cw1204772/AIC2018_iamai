# NVIDIA AI Challenge

Team name: iamai

## Detection

## Tracking

We use [iou-tracker](https://github.com/bochinski/iou-tracker) for tracking. To use it:

```
python3 Tracking/iou-tracker/demo.py -d <detection_csv> -o <output_csv> -sl 0.1 -sh 0.7 -si 0.5 -tm 2
```

## ReID

**Use in Single Camera Tracking**

```
python3 ReID/ReID.py <tracking_csv> <video> <output_csv> --window 15 --th 150
```

**Train on VeRi Dataset**

`cd ReID/ReID_CNN`

1. create train,query,gallery info .txt files

```
python3 create_VeRi_database.py [--img_dir <path_to_VeRi/image_train>] [--query_dir <path_to_VeRi/image_query>]
        [--gallery_dir <path_to_VeRi/image_test>] [--label_dir <path_to_VeRi/train_label.xml>]
        [--train_txt <VeRi_train_info.txt>][--query_txt <VeRi_query_info.txt>][--gallery_txt <VeRi_gallery_info.txt>]
```
2. train the model with VeRi or VeRi\_ict dataset
```
python3 train.py [--info <path to VeRi_train_info.txt>][--crop,flip,pretrain <True>][--lr <0.001>][--batch_size <64>]
                 [--n_epochs <20>][--save_model_dir <path_to_store_model>][--n_layer <18>(ResNet layer)]
                 [--dataset <VeRi or VeRi_ict>]
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
