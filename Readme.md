# NVIDIA AI Challenge

Team name: iamai

## ReID

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
  
  1. open matlab in the "VeRi\_cmc/" directory
  
  2. open "baseline\_evaluation\_FACT\_776.m" file
  
  3. change "dis\_CNN" mat path, "gt\_index",  "jk_index" txt file path
  
  4. run and get plot

