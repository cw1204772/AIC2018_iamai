import os
import glob
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import time as time
import models
import numpy as np



class VReID_Dataset(Dataset):
    def __init__(self, txt_file,resize=(224,224),crop=False,flip=False,pretrained_model=True,dataset='VeRi'):
        
        ######################
        self.img_list = []
        self.label_list = []
        self.color_list = []
        self.type_list = []
        self.dataset = dataset
        file = open(txt_file,'r')
        for row in file:
            line = row.strip().split(' ')
            self.img_list.append(line[0])
            self.label_list.append(int(line[1])-1)
            if self.dataset == 'VeRi_ict':
                self.color_list.append(int(line[2])-1)
                self.type_list.append(int(line[3])-1)
        file.close()
        self.n_id = len(set(self.label_list))
        
        if self.dataset=='VeRi_ict':
            self.n_color = len(set(self.color_list))
            self.n_type = len(set(self.type_list))
        else:
            self.n_color = 0
            self.n_type = 0
        index = np.random.permutation(len(self.label_list))
        # np.save('train_val_index.npy',index)
        # exit(-1)
        # print(self.n_id)
        index = np.load('train_val_index.npy')
        n_train = int(0.98*len(self.label_list))
        n_val = len(self.label_list)-n_train
        self.train_index = list(index[:n_train])
        self.val_index = list(index[-n_val:])
        self.n_train = len(self.train_index)
        self.n_val = len(self.val_index)

        #####################
        tran = []
        if crop == True:
            tran.append(transforms.Resize((resize[0]+50,resize[1]+50)))
            tran.append(transforms.RandomCrop(resize))
        else:
            tran.append(transforms.Resize(resize))
        if flip == True:
            tran.append(transforms.RandomHorizontalFlip())
            
        tran.append(transforms.ToTensor())
        if pretrained_model == True:
            tran.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]))
        self.compose = transforms.Compose(tran)
    def __getitem__(self,idx):
        img_name,label = self.img_list[idx],self.label_list[idx]
        img = Image.open(img_name)
        img = self.compose(img)
        if self.dataset == 'VeRi_ict':
            color = self.color_list[idx]
            type = self.type_list[idx]
            return{'img':img,'gt':torch.LongTensor([label]),'color':torch.LongTensor([color]),'type':torch.LongTensor([type])}
        return {'img':img,'gt':torch.LongTensor([label])}

    def __len__(self):
        return len(self.img_list)
            


def Get_train_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    sampler = SubsetRandomSampler(dataset.train_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=6)

def Get_val_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    sampler = SubsetRandomSampler(dataset.val_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=6)

# def generating_train_test_info():
    # label_list = []
    # file = open('train_info.txt','r')
    # file_content = file.readlines()
    # for row in file_content:
        # img,label = row.strip().split(' ')
        # label_list.append(label)
    # n_id = len(set(label_list))
    # test_id = list(np.random.permutation(n_id)[:int(0.3*n_id)])
    
    # train_label_dict = {}
    # test_label_dict = {}
    # tr_file = open('real_train_info.txt','w')
    # te_file = open('real_test_info.txt','w')
    # train_count = 0
    # test_count = 0
    # for row in file_content:
        # img,label = row.strip().split(' ')
        # if int(label) not in test_id:
            # if label not in train_label_dict:
                # train_label_dict[label]=train_count
                # train_count += 1
                # tr_file.write(img+' '+str(train_label_dict[label])+'\n')
            # else:
                # tr_file.write(img+' '+str(train_label_dict[label])+'\n')
        # else:
            # if label not in test_label_dict:
                # test_label_dict[label]=test_count
                # test_count +=1
                # te_file.write(img+' '+str(test_label_dict[label])+'\n')
            # else:
                # te_file.write(img+' '+str(test_label_dict[label])+'\n')

if __name__ == '__main__':
    
    pretrained = True

    D = VReID_Dataset(sys.argv[1],crop=True,flip=True,pretrained_model=pretrained)
    loader = Get_DataLoader(D)

    print('len:',len(D))
    print('n_id:',D.n_id)
    print('n_batch:',len(D)//128+1)

    




