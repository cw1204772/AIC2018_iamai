# Adaptive Feature Learning CNN

Here, we will train the CNN feature extractor for our vehicle Re-ID system.  
This step should be performed after "Post-Tracking" stage of the system.  
First, download VeRi, CompCars, BoxCars116k dataset.  
Then, run the following script to setup training:
```
bash setup.sh <VeRi_DIR> <WORK_DIR> <CompCars_DIR> <BoxCars116k_DIR>
```
(`<WORK_DIR>` is the working directory you setup for the system)  
Now, we are ready to train!

## Joint Training (AFL)

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

## Train on VeRi Dataset

* Train classification model with VeRi or VeRi\_ict dataset
```
python3 train.py --info VeRi_train_info.txt --lr 0.001 --batch_size 64 --n_epochs 20 --n_layer 18 --dataset VeRi
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
