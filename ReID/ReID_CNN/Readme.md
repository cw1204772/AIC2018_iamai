# Adaptive Feature Learning CNN

Here, we will train the CNN feature extractor for our vehicle Re-ID system.
We introduce the adaptive feature learning (AFL) technique to alleviate the requirements of labeled videos in the testing environment.
It is based on the observation that one vehicle can not appear at multiple locations at the same time, and it should move continuously along the time.
We call this the space-time prior and illustate it in the figure below.
We exploit this nature in traffic videos to generate triplets, along with samples from existing vehicle datasets([VeRi](https://github.com/VehicleReId/VeRidataset), [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html), [BoxCars116k](https://medusa.fit.vutbr.cz/traffic/research-topics/fine-grained-vehicle-recognition/boxcars-improving-vehicle-fine-grained-recognition-using-3d-bounding-boxes-in-traffic-surveillance/)), to train our CNN in a multi-task learning manner.

More detail can be found in our paper:  
Chih-Wei Wu, Chih-Ting Liu, Chen-En Jiang, Wei-Chih Tu, Shao-Yi Chien "Vehicle Re-Identification with the Space-Time Prior" CVPRW, 2018.  

![adaptive_feature_learning](https://github.com/cw1204772/AIC2018_iamai/raw/master/ReID/ReID_CNN/afl4.png "The space-time prior exploit for adaptive feature learning")

The following instruction are for training CNN by yourself.  
We also provide our own model weight here \[[link](https://drive.google.com/open?id=1M-V-TilFg5yyVRsySCTOoFGgcn1HYlY3)\].

## Train on Multiple Datasets

Please follow the steps below for training CNN:
1. Install requirements in root system's [Readme.md](https://github.com/cw1204772/AIC2018_iamai#requirements).
2. In this step, we prepare data for [2018 NVIDIA AI City Challenge](https://www.aicitychallenge.org/). Follow detail guide section in root system's [Readme.md](https://github.com/cw1204772/AIC2018_iamai#detail-guide) until finishing stage III, Post-Tracking.
3. Download and extract [VeRi](https://github.com/VehicleReId/VeRidataset), [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html), [BoxCars116k](https://medusa.fit.vutbr.cz/traffic/research-topics/fine-grained-vehicle-recognition/boxcars-improving-vehicle-fine-grained-recognition-using-3d-bounding-boxes-in-traffic-surveillance/) dataset.  
4. Run the following script to setup training:
```
bash setup.sh <VeRi_DIR> <WORK_DIR> <CompCars_DIR> <BoxCars116k_DIR>
```
(`<WORK_DIR>` is the working directory you setup for the system in step 2)  
5. Now, we are ready to train! Train the model by:
```
python3 train_joint.py --veri_txt VeRi_train_info.txt \
                       --compcars_txt Comp_info/Comp_sv_train.txt \
                       --compcars_test_txt Comp_info/Comp_sv_test.txt \
                       --boxcars_txt BoxCars_train.txt \
                       --boxcars_test_txt BoxCars_test.txt \
                       --aic_pkl AIC2018.pkl \
                       --n_epochs 1501 --save_model_dir ./ckpt \
                       --n_layer 50 --lr 1e-5 \
                       --margin soft --batch_hard \
                       --save_every_n_epoch 10 \
                       --class_w 1 --batch_size 128 \
                       --class_in_batch 32
```
The model will be in `./ckpt`

## Train on VeRi Dataset

If your are lazy to prepare so many datasets, we provide instructions for training only on VeRi dataset.  
* Train classification model with VeRi or VeRi\_ict dataset
```
python3 train.py --info VeRi_train_info.txt --save_model_dir ./ckpt --lr 0.001 --batch_size 64 --n_epochs 20 --n_layer 18 --dataset VeRi
```

* Train triplet model with VeRi dataset
```
python3 train.py --info VeRi_train_info.txt --n_epochs 1500 --save_model_dir ./ckpt --n_layer 18 --margin soft --class_in_batch 32 --triplet --lr 0.001 --batch_hard --save_every_n_epoch 50
```

## Evaluate CNN on VeRi

1. dump distance matrix
```
python3 compute_VeRi_dis.py --load_ckpt <base_model_path> --n_layer <Resnet_layer> --gallery_txt VeRi_gallery_info.txt --query_txt VeRi_query_info.txt --dis_mat dist_CNN.mat
```

2. compute cmc curve
```  
  1. open matlab in the "VeRi_cmc/" directory
  2. open "baseline_evaluation_FACT_776.m" file
  3. change "dis_CNN" mat path, "gt_index",  "jk_index" txt file path
  4. run and get plot
```

## Reference
* NVIDIA AI City Challenge. https://www.aicitychallenge.org/, 2018.
* X. Liu, W. Liu, H. Ma, and H. Fu. Large-scale vehicle reidentification in urban surveillance videos. ICME, 2016.
* X. Liu, W. Liu, T. Mei, and H. Ma. A deep learning-based approach to progressive vehicle re-identification for urban surveillance. ECCV, 2016
* L. Yang, P. Luo, C. C. Loy, and X. Tang. A large-scale car dataset for fine-grained categorization and verification. CVPR, 2015.
* J. Sochor, J. pahel, and A. Herout. Boxcars: Improving finegrained recognition of vehicles using 3-d bounding boxes in traffic surveillance. IEEE Transactions on Intelligent Transportation Systems, PP(99):1–12, 2018.

## Citing

```
@inproceedings{wu2018vreid,
  title={Vehicle Re-Identification with the Space-Time Prior},
  author={Wu, Chih-Wei and Liu, Chih-Ting and Jiang, Chen-En and Tu, Wei-Chih and Chien, Shao-Yi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshop},
  year={2018},
}
```
