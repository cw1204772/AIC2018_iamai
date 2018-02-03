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
    def __init__(self, txt_file,resize=(224,224),crop=False,flip=False,jitter=0,pretrained_model=True,dataset='VeRi'):
        
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
        n_train = int(0.95*len(self.label_list))
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
        if jitter != 0:
            tran.append(transforms.ColorJitter(brightness=jitter))
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
            
class TripletImage_Dataset(Dataset):
    def __init__(self, db_txt, resize=(224,224), crop=False, flip=False, jitter=0, 
                 imagenet_normalize=True, val_split=0.01, 
                 class_in_batch=32, image_per_class_in_batch=4):

        # Load image list, class list
        txt = np.loadtxt(db_txt, dtype=str)
        self.imgs = txt[:, 0]
        self.classes, self.n_id = Remap_Label(txt[:, 1].astype(int))
        if not Check_Min_Sample_Per_Class(self.classes, image_per_class_in_batch): 
            return ValueError('There is not enough samples per class! (Min {} samples required)'\
                              .format(image_per_class_in_batch))
        self.len = self.n_id
        self.class_in_batch = class_in_batch
        self.image_per_class_in_batch = image_per_class_in_batch

        # Validation split (split according to id)
        permute_idx = np.random.permutation(self.n_id)
        self.val_index = permute_idx[:int(val_split*self.n_id)]
        self.train_index = permute_idx[int(val_split*self.n_id):]
        self.n_train = int(self.len * (1-val_split))
        self.n_val = int(self.len * val_split)

        # Transform
        trans = []
        if crop:
            trans.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans.append(transforms.RandomCrop(resize))
        else:
            trans.append(transforms.Resize(resize))
        if flip: trans.append(transforms.RandomHorizontalFlip())
        if jitter: trans.append(transforms.ColorJitter(brightness=jitter))
        trans.append(transforms.ToTensor())
        if imagenet_normalize:
            trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(trans)
        
    def __getitem__(self, idx):
        id = torch.arange(self.n_id).long()[idx]
        final_idx = np.zeros((self.image_per_class_in_batch,))
        select = np.nonzero(self.classes == id)[0]
        select = np.random.permutation(select)[:self.image_per_class_in_batch]
        output = {'img':[], 'class':[]}
        for i in select.tolist():
            img = Image.open(self.imgs[i])
            output['img'].append(self.transform(img).unsqueeze(0))
            output['class'].append(id)
        output['img'] = torch.cat(output['img'], dim=0)
        output['class'] = torch.LongTensor(output['class'])
        return output

    def __len__(self):
        return self.len

def Get_train_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    sampler = SubsetRandomSampler(dataset.train_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=6)

def Get_val_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    sampler = SubsetRandomSampler(dataset.val_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=6)

def Remap_Label(labels):
    labels = labels - np.min(labels)
    unique_label = np.unique(labels)
    label_map = np.zeros(np.max(unique_label)+1, dtype=int)
    for i, l in enumerate(unique_label.tolist()):
        label_map[l] = i
    return label_map[labels], len(unique_label)

def Check_Min_Sample_Per_Class(labels, min):
    unique_labels = np.unique(labels)
    for i in unique_labels.tolist():
        if (labels == i).sum() < min:
            return False
    return True

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
    '''
    pretrained = True

    D = VReID_Dataset(sys.argv[1],crop=True,flip=True,pretrained_model=pretrained)
    loader = Get_DataLoader(D)

    print('len:',len(D))
    print('n_id:',D.n_id)
    print('n_batch:',len(D)//128+1)
    '''
    #labels = (np.arange(10)+1)*10
    #print(labels)
    #labels = Remap_Label(labels)
    #print(labels)

    dataset = TripletImage_Dataset(sys.argv[1], crop=True, flip=True, jitter=5, imagenet_normalize=True)
    train_loader = Get_train_DataLoader(dataset, batch_size=32, num_workers=1)
    val_loader = Get_val_DataLoader(dataset, batch_size=32)
    print('len', len(dataset))
    print('train n_batch', len(train_loader))
    print('val n_batch', len(val_loader))
    for data in train_loader:
       print(data.keys())
       print(data['img'][0].size())
       print(data['class'][0].size())
       exit(-1)
    for data in val_loader:
       print(data.keys())
       print(data['img'])
       print(data['class'])


