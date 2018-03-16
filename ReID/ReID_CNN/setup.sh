VeRi_dir=$1
comp_dir=$2
sv_dir=$3
python3 create_VeRi_database.py --img_dir $VeRi_dir/image_train --query_dir $VeRi_dir/image_query --gallery_dir $VeRi_dir/image_test --label_dir $VeRi_dir/train_label.xml --train_txt VeRi_train_info.txt --query_txt VeRi_query_info.txt --gallery_txt VeRi_gallery_info.txt
python3 create_Comp_database.py --sv_dataset_dir $sv_dir --comp_dataset_dir $comp_dir --info_dir Comp_info  
