import os
import glob
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import Compose,Resize,ToTensor
from PIL import Image,ImageEnhance
import time as time
import models
import numpy as np
import torchvision.utils as vutils
import pickle
import pandas as pd

def load_dataset(db_txt):
    data = pd.read_csv(db_txt, sep=' ').to_dict('list')
    return data

class BoxCars_Dataset(Dataset):
    def __init__(self, db_txt, resize=(224,224), crop=False, flip=False, jitter=False, 
                 imagenet_normalize=True, val_split=0.01 ):
        # Load image list, class list
        txt = np.loadtxt(db_txt, dtype=str)[1:]
        self.imgs = txt[:, 0]
        self.models, self.n_models = Remap_Label(txt[:, 1].astype(int))
        self.bboxs = txt[:,2:6].astype(int)
        self.len = len(self.imgs)
        # Validation split
        permute_idx = np.random.permutation(self.len)
        self.val_index = permute_idx[:int(val_split*self.len)]
        self.train_index = permute_idx[int(val_split*self.len):]
        #transform
        self.jitter = jitter
        trans_PIL = []
        if crop:
            trans_PIL.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans_PIL.append(transforms.RandomCrop(resize))
        else:
            trans_PIL.append(transforms.Resize(resize))
        if flip: trans_PIL.append(transforms.RandomHorizontalFlip())
        trans_Tensor = []
        if not self.jitter: 
            trans_Tensor.append(transforms.ToTensor())
        if imagenet_normalize:
            trans_Tensor.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_PIL = transforms.Compose(trans_PIL)
        self.transform_Tensor = transforms.Compose(trans_Tensor)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        model = self.models[idx]
        x, y, w, h = self.bboxs[idx]
        img = Image.open(img_name)
        img = img.crop((x, y, x+w, y+h))
        if self.jitter:
            img = self.transform_PIL(img)
            img = Jitter_Transform_to_Tensor(img)
            img = self.transform_Tensor(img)
        else:
            img = self.transform_Tensor(self.transform_PIL(img))
        return {'img':img, 'model':int(model)}
    def __len__(self):
        return self.len

class sv_comp_Dataset(Dataset):
    def __init__(self, db_txt, resize=(224,224), crop=False, flip=False, jitter=False, 
                 imagenet_normalize=True, val_split=0.01 ):
        # Load image list, class list
        txt = np.loadtxt(db_txt, dtype=str)[1:]
        self.imgs = txt[:, 0]
        self.makes, self.n_makes = Remap_Label(txt[:, 1].astype(int))
        self.models, self.n_models = Remap_Label(txt[:, 2].astype(int))
        # don't use remap on colors
        self.colors  = txt[:,3].astype(int)
        self.n_colors = 12
        self.len = len(self.makes)
        # Validation split
        permute_idx = np.random.permutation(self.len)
        self.val_index = permute_idx[:int(val_split*self.len)]
        self.train_index = permute_idx[int(val_split*self.len):]
        self.n_train = int(self.len * (1-val_split))
        self.n_val = int(self.len * val_split)
        #transform
        self.jitter = jitter
        trans_PIL = []
        if crop:
            trans_PIL.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans_PIL.append(transforms.RandomCrop(resize))
        else:
            trans_PIL.append(transforms.Resize(resize))
        if flip: trans_PIL.append(transforms.RandomHorizontalFlip())
        trans_Tensor = []
        if not self.jitter: 
            trans_Tensor.append(transforms.ToTensor())
        if imagenet_normalize:
            trans_Tensor.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_PIL = transforms.Compose(trans_PIL)
        self.transform_Tensor = transforms.Compose(trans_Tensor)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        make = self.makes[idx]
        model = self.models[idx]
        color = self.colors[idx]
        img = Image.open(img_name)
        rng = np.random.randint(70,224)
        img = img.resize((rng,rng))
        if self.jitter:
            img = self.transform_PIL(img)
            img = Jitter_Transform_to_Tensor(img)
            img = self.transform_Tensor(img)
        else:
            img = self.transform_Tensor(self.transform_PIL(img))
        return {'img':img,'make':int(make),'model':int(model),'color':int(color)}
    def __len__(self):
        return self.len

class comp_Dataset(Dataset):
    def __init__(self, db_txt, resize=(224,224), crop=False, flip=False, jitter=False, 
                 imagenet_normalize=True, val_split=0.01 ):
        # Load image list, class list
        flip = False
        txt = np.loadtxt(db_txt, dtype=str)[1:]
        self.imgs = txt[:, 0]
        self.makes, self.n_makes = Remap_Label(txt[:, 1].astype(int))
        self.models, self.n_models = Remap_Label(txt[:, 2].astype(int))
        self.angle, self.n_angle = Remap_Label(txt[:, 3].astype(int))
        self.crop_4  = txt[:,4:].astype(int)
        self.len = len(self.makes)
        # Validation split 
        permute_idx = np.random.permutation(self.len)
        self.val_index = permute_idx[:int(val_split*self.len)]
        self.train_index = permute_idx[int(val_split*self.len):]
        self.n_train = int(self.len * (1-val_split))
        self.n_val = int(self.len * val_split)
        #transform
        self.jitter = jitter
        trans_PIL = []
        if crop:
            trans_PIL.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans_PIL.append(transforms.RandomCrop(resize))
        else:
            trans_PIL.append(transforms.Resize(resize))
        if flip: trans_PIL.append(transforms.RandomHorizontalFlip())
        trans_Tensor = []
        if not self.jitter: 
            trans_Tensor.append(transforms.ToTensor())
        if imagenet_normalize:
            trans_Tensor.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_PIL = transforms.Compose(trans_PIL)
        self.transform_Tensor = transforms.Compose(trans_Tensor)
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        make = self.makes[idx]
        model = self.models[idx]
        angle = self.angle[idx]
        x1 = self.crop_4[idx][0]
        y1 = self.crop_4[idx][0]
        w = self.crop_4[idx][2]-self.crop_4[idx][0]
        h = self.crop_4[idx][3]-self.crop_4[idx][1]
        img = Image.open(img_name)
        img = img.crop((x1,y1,w,h))
        rng = np.random.randint(70,224)
        img = img.resize((rng,rng))
        if self.jitter:
            img = self.transform_PIL(img)
            img = Jitter_Transform_to_Tensor(img)
            img = self.transform_Tensor(img)
        else:
            img = self.transform_Tensor(self.transform_PIL(img))
        return {'img':img,'make':int(make),'model':int(model),'angle':int(angle)}
    def __len__(self):
        return self.len

class VReID_Dataset(Dataset):
    def __init__(self, txt_file,resize=(224,224),crop=False,flip=False,jitter=0,pretrained_model=True,dataset='VeRi'):
        
        ######################
        self.img_list = []
        self.label_list = []
        self.color_list = []
        self.type_list = []
        self.dataset = dataset
        file = open(txt_file,'r')
        for i,row in enumerate(file):
            if i==0: continue
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
        rng = np.random.randint(70,224)
        img = img.resize((rng,rng))
        img = self.compose(img)
        if self.dataset == 'VeRi_ict':
            color = self.color_list[idx]
            type = self.type_list[idx]
            return{'img':img,'gt':torch.LongTensor([label]),'color':torch.LongTensor([color]),'type':torch.LongTensor([type])}
        return {'img':img,'gt':torch.LongTensor([label])}

    def __len__(self):
        return len(self.img_list)
            
class TripletImage_Dataset(Dataset):
    def __init__(self, db_txt, resize=(224,224), crop=False, flip=False, jitter=False, 
                 imagenet_normalize=True, val_split=0.01, 
                 class_in_batch=32, image_per_class_in_batch=4):

        # Load image list, class list
        txt = np.loadtxt(db_txt, dtype=str)[1:]
        self.imgs = txt[:, 0]
        self.classes, self.n_id = Remap_Label(txt[:, 1].astype(int))
        self.colors, self.n_colors = Remap_Label(txt[:,2].astype(int))
        self.n_colors = 12

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
        self.jitter = jitter
        trans_PIL = []
        if crop:
            trans_PIL.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans_PIL.append(transforms.RandomCrop(resize))
        else:
            trans_PIL.append(transforms.Resize(resize))
        if flip: trans_PIL.append(transforms.RandomHorizontalFlip())
        trans_Tensor = []
        if not self.jitter: 
            trans_Tensor.append(transforms.ToTensor())
        if imagenet_normalize:
            trans_Tensor.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_PIL = transforms.Compose(trans_PIL)
        self.transform_Tensor = transforms.Compose(trans_Tensor)
        
    def __getitem__(self, idx):
        # for id
        id = np.arange(self.n_id)[idx]
        final_idx = np.zeros((self.image_per_class_in_batch,))
        select = np.nonzero(self.classes == id)[0]
        select = np.random.permutation(select)[:self.image_per_class_in_batch]
        # for color
        color = int(self.colors[select][0])
        output = {'img':[], 'class':[],'color':[]}
        for i in select.tolist():
            img = Image.open(self.imgs[i])
            rng = np.random.randint(70,224)
            img = img.resize((rng,rng))
            if self.jitter:
                img = self.transform_PIL(img)
                img = Jitter_Transform_to_Tensor(img)
                img = self.transform_Tensor(img)
            else:
                img = self.transform_Tensor(self.transform_PIL(img))
            output['img'].append(img.unsqueeze(0))
            output['class'].append(id)
            output['color'].append(color)
        output['img'] = torch.cat(output['img'], dim=0)
        output['class'] = torch.LongTensor(output['class'])
        output['color'] = torch.LongTensor(output['color'])
        return output

    def __len__(self):
        return self.len

class Unsupervised_TripletImage_Dataset(Dataset):
    def __init__(self, pkl_name, resize=(224,224), crop=False, flip=False, jitter=False, 
                 imagenet_normalize=True, batch_size=128):

        # Load sample list, image list
        with open(pkl_name, 'rb') as f:
            data = pickle.load(f)
        self.samples = data['samples']
        self.imgs = data['track_imgs']
        self.batch_size = batch_size
        self.len = sum([len(s) for s in self.samples]) // batch_size
        self.train_index = list(range(self.len))
        self.n_id = 1

        # Transform
        self.jitter = jitter
        trans_PIL = []
        if crop:
            trans_PIL.append(transforms.Resize((resize[0]+50, resize[1]+50)))
            trans_PIL.append(transforms.RandomCrop(resize))
        else:
           trans_PIL.append(transforms.Resize(resize))
        if flip: trans_PIL.append(transforms.RandomHorizontalFlip())
        trans_Tensor = []
        if not self.jitter: 
            trans_Tensor.append(transforms.ToTensor())
        if imagenet_normalize:
            trans_Tensor.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_PIL = transforms.Compose(trans_PIL)
        self.transform_Tensor = transforms.Compose(trans_Tensor)
        
    def __getitem__(self, idx):
        np.random.seed(time.time())
        sample_i = np.random.randint(len(self.samples))
        track_i = 0
        img_i = 0
        rng = np.random.permutation(len(self.imgs[self.samples[sample_i][track_i]]))

        output_img = torch.zeros(self.batch_size, 3, 224, 224)
        sample_record = -1*np.ones((self.batch_size,))
        track_record = -1*np.ones((self.batch_size,))
        #img_record = -1*np.ones((self.batch_size,))
        for i in range(self.batch_size):
            img_name = None
            while img_name is None:
                if track_i < len(self.samples[sample_i]):
                    if img_i < len(self.imgs[self.samples[sample_i][track_i]]) and img_i < len(rng):
                        img_name = self.imgs[self.samples[sample_i][track_i]][rng[img_i]]
                        sample_record[i] = sample_i
                        track_record[i] = track_i
                        #img_record[i] = rng[img_i]
                        img_i += 1
                    elif track_i < len(rng):
                        track_i += 1
                        img_i = 0
                        if track_i < len(self.samples[sample_i]): 
                            rng = np.random.permutation(len(self.imgs[self.samples[sample_i][track_i]]))
                    else:
                        track_i += 1
                        img_i = 0
                        if track_i < len(self.samples[sample_i]):
                            rng = np.random.permutation(len(self.imgs[self.samples[sample_i][track_i]]))
                else:
                    sample_i = np.random.randint(len(self.samples))
                    track_i = 0
                    img_i = 0
                    rng = np.random.permutation(len(self.imgs[self.samples[sample_i][track_i]]))

            # Read image
            img = Image.open(img_name)
            if self.jitter:
                img = self.transform_PIL(img)
                img = Jitter_Transform_to_Tensor(img)
                img = self.transform_Tensor(img)
            else:
                img = self.transform_Tensor(self.transform_PIL(img))
            output_img[i, :, :, :] = img

        #print('sample_record:', sample_record)
        #print('track_record:', track_record)
        #print('img_record:', img_record)
        #print(list(zip(sample_record.tolist(), track_record.tolist(), img_record.tolist())))
        sample_record = sample_record.reshape(-1, 1)
        sample_record = sample_record == sample_record.T
        track_record = track_record.reshape(-1, 1)
        track_record = track_record == track_record.T
        pos_mask = (sample_record & track_record) ^ np.eye(self.batch_size,dtype=bool)
        neg_mask = sample_record & (~track_record)

        output = {'img':output_img,
                  'pos_mask':torch.from_numpy(pos_mask.astype(np.uint8)),
                  'neg_mask':torch.from_numpy(neg_mask.astype(np.uint8))}
        return output

    def __len__(self):
        return self.len

def Get_test_DataLoader(dataset,batch_size=128,shuffle=False,num_workers=6):
    return DataLoader(dataset,batch_size = batch_size,shuffle=shuffle,num_workers=num_workers)

def Get_train_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    sampler = SubsetRandomSampler(dataset.train_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=num_workers)

def Get_val_DataLoader(dataset,batch_size=128,shuffle=False,num_workers=6):
    sampler = SubsetRandomSampler(dataset.val_index)
    return DataLoader(dataset,batch_size = batch_size,sampler=sampler,num_workers=num_workers)

def Remap_Label(labels):
    labels = labels - np.min(labels)
    unique_label = np.unique(labels)
    label_map = np.zeros(np.max(unique_label)+1, dtype=int)
    for i, l in enumerate(unique_label.tolist()):
        label_map[l] = i
    return label_map[labels], len(unique_label)
# def Remap_Color(colors):

def Check_Min_Sample_Per_Class(labels, min):
    unique_labels = np.unique(labels)
    for i in unique_labels.tolist():
        if (labels == i).sum() < min:
            return False
    return True


def Jitter_Transform_to_Tensor(img_):
    coin = np.random.uniform(0,1)
    coin = 0.9
    if coin> 0.4 and coin < 0.7: #normal
        i_factor = 1
        c_factor = 1
        r_factor = 1
        b_factor = 1
    elif coin <= 0.2: #high shadow blue
        i_factor = 0.6
        c_factor = 0.8
        r_factor = 0.9
        b_factor = 1.3
    elif coin > 0.2 and coin <= 0.4: #little shadow blue
        i_factor = 0.7
        c_factor = 0.7
        r_factor = 0.9
        b_factor = 1.2
    elif coin >= 0.7 and coin<0.8: #little bright
        i_factor = 1.2
        c_factor = 1.2
        r_factor = 1
        b_factor = 1
    elif coin >= 0.8 and coin<0.9: # little yellow
        i_factor = 0.7
        c_factor = 0.7
        r_factor = 1.2
        b_factor = 0.9
    elif coin >= 0.9 and coin <1: # higher shadow yellow
        i_factor = 0.6
        c_factor = 0.6
        r_factor = 1.4
        b_factor = 0.8
    enhancer_br = ImageEnhance.Brightness(img_)
    img = enhancer_br.enhance(i_factor)
    enhancer_con = ImageEnhance.Contrast(img)
    img = enhancer_con.enhance(c_factor)

    totensor = transforms.ToTensor()
    img = totensor(img)
    img[0,:,:] *=r_factor
    img[2,:,:] *=b_factor
    return img    
    
if __name__ == '__main__':
    re = transforms.Resize((224,224))
    '''This part
    totensor = transforms.ToTensor()
    brightness_factor = float(sys.argv[2])
    contrast_factor = float(sys.argv[3])
    hue_factor = float(sys.argv[4])

    img = Image.open(sys.argv[1])
    img = re(img)

    enhancer_br = ImageEnhance.Brightness(img)
    img = enhancer_br.enhance(brightness_factor)
    enhancer_con = ImageEnhance.Contrast(img)
    img = enhancer_con.enhance(contrast_factor)
    img = totensor(img)
    vutils.save_image(img,'testimg.jpg')
    exit(-1)
    '''

    '''And This part
    img = Image.open(sys.argv[1])
    re = transforms.Resize((224,224))
    img = re(img)
    img = Jitter_Transform_to_Tensor(img)
    vutils.save_image(img,'testimg.jpg')
    exit(-1)
    # '''
    # '''
    # pretrained = True

    # D = VReID_Dataset(sys.argv[1],crop=True,flip=True,pretrained_model=pretrained)
    # loader = Get_DataLoader(D)

    # print('len:',len(D))
    # print('n_id:',D.n_id)
    # print('n_batch:',len(D)//128+1)
    #labels = (np.arange(10)+1)*10
    #print(labels)
    #labels = Remap_Label(labels)
    #print(labels)

    torch.set_printoptions(threshold=5000)
    #dataset = Unsupervised_TripletImage_Dataset(sys.argv[1], crop=True, flip=True, jitter=True, imagenet_normalize=True, batch_size=32, image_per_class=2)
    dataset = sv_comp_Dataset(sys.argv[1], crop=True, flip=True, jitter=False, imagenet_normalize=False)
    train_loader = Get_train_DataLoader(dataset, batch_size=1, num_workers=1)
    #val_loader = Get_val_DataLoader(dataset, batch_size=32)
    #dataset = comp_Dataset(sys.argv[1],crop=True,flip=True,jitter=True)
    #train_loader = Get_train_DataLoader(dataset, batch_size=32, num_workers=1)
    #val_loader = Get_val_DataLoader(dataset, batch_size=32)

    # dataset = TripletImage_Dataset(sys.argv[1], crop=True, flip=True, jitter=True, imagenet_normalize=True)
    # train_loader = Get_train_DataLoader(dataset, batch_size=32, num_workers=1)
    # val_loader = Get_val_DataLoader(dataset, batch_size=32)
    print('len', len(dataset))
    print('train n_batch', len(train_loader))
    #print('val n_batch', len(val_loader))
    for data in train_loader:
       print(data.keys())
       print(data['img'].size())
       #print(data['pos_mask'])
       #print(data['neg_mask'])
       #print(data['make'].size())
       print(data['model'].size())
       img = transforms.ToPILImage()(data['img'][0])
       img.save('temp.jpg')
       #print(data['angle'].size())
       exit(-1)
    #for data in val_loader:
    #   print(data.keys())
    #   print(data['img'])
    #   print(data['class'])
    # '''


