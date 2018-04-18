import scipy.io as sio
import os
import argparse
import numpy as np
import pathlib

def Remap_Label(labels):
    label_dict = {}
    unique_label = np.unique(labels)
    for i,j in enumerate(unique_label):
        label_dict[j[0]]=i
    for i,j in enumerate(labels):
        labels[i] = label_dict[j[0]]
    return labels,label_dict
def alter_color_to_VeRi(color):
    if color == 0:
        return 9
    elif color == 1:
        return 6
    elif color == 2:
        return 4
    elif color == 3:
        return 0
    elif color == 4:
        return 5
    elif color == 5:
        return 2
    elif color == 6:
        return 10
    elif color == 7:
        return 8
    elif color == 8:
        return 7
    elif color == 9:
        return 3
    elif color == -1:
        return 11


if __name__ == '__main__':
  # Argparse
    parser = argparse.ArgumentParser(description='Database generator for UA-DETRAC dataset')
    parser.add_argument('--sv_dataset_dir', help='the dir containing sv dataset')
    parser.add_argument('--comp_dataset_dir', help='the dir containing comp dataset ')
    parser.add_argument('--sv_image_dir', help='the dir containing sv dataset(image)')
    parser.add_argument('--comp_image_dir', help='the dir containing comp dataset(image)')
    parser.add_argument('--info_dir', help='output dir')
    args = parser.parse_args()
    os.system('mkdir -p %s'%(args.info_dir))   


    make_model_label = sio.loadmat(os.path.join(args.sv_dataset_dir,'sv_make_model_name.mat'))['sv_make_model_name']
    make_model_label = make_model_label[:,:2]
    make_model_label[:,0],make_dict = Remap_Label(make_model_label[:,0])
    make_model_label[:,1],model_dict = Remap_Label(make_model_label[:,1])

    color_label = sio.loadmat(os.path.join(args.sv_dataset_dir,'color_list.mat'))['color_list']
    # create color list
    color_dict = {}
    for i in range(color_label.shape[0]):
        color_dict[color_label[i][0][0]] = alter_color_to_VeRi(color_label[i][1][0][0]) 
    # open train/test split
    file_train = open(os.path.join(args.sv_dataset_dir,'train_surveillance.txt'),'r')
    file_test = open(os.path.join(args.sv_dataset_dir,'test_surveillance.txt'),'r')
    # train
    file = open(os.path.join(args.info_dir,'Comp_sv_train.txt'),'w')
    file.write('img_path make model color\n')
    for row in file_train:
        id,file_name = row.strip().split('/')
        path_name = os.path.join(args.sv_image_dir,'image',row.strip())
        path_name = str(pathlib.Path(path_name).resolve())
        make = str(make_model_label[int(id)-1][0])
        model = str(make_model_label[int(id)-1][1])
        color = str(color_dict[row.strip()])
        file.write('%s %s %s %s\n'%(path_name,make,model,color))
    file.close()
    #test
    file = open(os.path.join(args.info_dir,'Comp_sv_test.txt'),'w')
    file.write('img_path make model color\n')
    for row in file_test:
        id,file_name = row.strip().split('/')
        path_name = os.path.join(args.sv_image_dir,'image',row.strip())
        path_name = str(pathlib.Path(path_name).resolve())
        make = str(make_model_label[int(id)-1][0])
        model = str(make_model_label[int(id)-1][1])
        color = str(color_dict[row.strip()])
        file.write('%s %s %s %s\n'%(path_name,make,model,color))
    file.close()
    file_train.close()
    file_test.close()

    #output info
    file = open(os.path.join(args.info_dir,'Comp_sv_make_dict.txt'),'w')
    keys = np.array(list(make_dict.keys()))
    values = np.array(list(make_dict.values()))
    idx = np.argsort(values)
    keys = keys[idx]
    values = values[idx]
    for i in range(keys.shape[0]):
        file.write(str(values[i])+' '+str(keys[i])+'\n')
    file.close()
    
    file = open(os.path.join(args.info_dir,'Comp_sv_model_dict.txt'),'w')
    keys = np.array(list(model_dict.keys()))
    values = np.array(list(model_dict.values()))
    idx = np.argsort(values)
    keys = keys[idx]
    values = values[idx]
    for i in range(keys.shape[0]):
        file.write(str(values[i])+' '+str(keys[i])+'\n')
    file.close()
    #################################################################################################
    file_train = open(os.path.join(args.comp_dataset_dir,'train_test_split','classification','train.txt'),'r')
    file_test = open(os.path.join(args.comp_dataset_dir,'train_test_split','classification','test.txt'),'r')
    file = open(os.path.join(args.info_dir,'Comp_train.txt'),'w')
    file.write('img_path make model angle bbox_x1 bbox_y1 bbox_x2 bbox_y2\n')
    for row in file_train:
        row = row.strip()
        file_path = os.path.join(args.comp_image_dir,'image',row)
        path_name = str(pathlib.Path(path_name).resolve())
        label_file = open(os.path.join(args.comp_dataset_dir,'label',row.replace('jpg','txt')),'r')
        angle,_,co = label_file.readlines()
        angle=str(0) if int(angle.strip())==-1 else str(angle.strip())
        coordinate = co.strip()
        make = row.split('/')[0]
        model = row.split('/')[1]
        file.write('%s %s %s %s %s\n'%(file_path,make,model,angle,coordinate))
    file= open(os.path.join(args.info_dir,'Comp_test.txt'),'w')
    file.write('img_path make model angle bbox_x1 bbox_y1 bbox_x2 bbox_y2\n')
    for row in file_test:
        row = row.strip()
        file_path = os.path.join(args.comp_image_dir,'image',row)
        path_name = str(pathlib.Path(path_name).resolve())
        label_file = open(os.path.join(args.comp_dataset_dir,'label',row.replace('jpg','txt')),'r')
        angle,_,co = label_file.readlines()
        angle=str(0) if int(angle.strip())==-1 else str(angle.strip())
        coordinate = co.strip()
        make = row.split('/')[0]
        model = row.split('/')[1]
        file.write('%s %s %s %s %s\n'%(file_path,make,model,angle,coordinate))
    file.close()
    file_train.close()
    file_test.close()
    label_file.close()






