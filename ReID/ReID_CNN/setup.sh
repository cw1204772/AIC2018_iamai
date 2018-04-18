VeRi_dir=$1
AIC_dir=$2
COMPCARS_DIR=$3
BOXCARS_DIR=$4
DATABASE_DIR=.
mkdir $DATABASE_DIR
python3 create_VeRi_database.py --img_dir $VeRi_dir/image_train --query_dir $VeRi_dir/image_query \
                                --gallery_dir $VeRi_dir/image_test --label_dir $VeRi_dir/train_label.xml \
                                --train_txt $DATABASE_DIR/VeRi_train_info.txt \
                                --query_txt $DATABASE_DIR/VeRi_query_info.txt \
                                --gallery_txt $DATABASE_DIR/VeRi_gallery_info.txt
python3 create_Comp_database.py --comp_dataset_dir $COMPCARS_DIR/data --sv_dataset_dir $COMPCARS_DIR/sv_data \
                                --comp_image_dir $COMPCARS_DIR/data --sv_image_dir $COMPCARS_DIR/sv_data \
                                --info_dir $DATABASE_DIR/Comp_info
python3 create_AIC_database.py $AIC_dir/post_tracking/fasta  --output_pkl $DATABASE_DIR/AIC2018.pkl
python3 create_BoxCars_database.py --dataset_dir $BOXCARS_DIR --image_dir $BOXCARS_DIR --output_dir $DATABASE_DIR
