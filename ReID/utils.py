import os
import glob
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import time as time
import models



class V_Re_ID_Dataset(Dataset):
    def __init__(self, txt_file,resize=(224,224),crop=False,flip=False,pretrained_model=True):
        self.t_resize = transforms.Resize(resize)
        if crop == True:
            self.t_b_resize = transforms.Resize((resize[0]+50,resize[1]+50))
            self.t_crop = transforms.RandomCrop(resize)
            self.n_crop = 3
        if flip == True:
            self.t_flip = transforms.RandomHorizontalFlip()
        if pretrained_model == True:
            self.t_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.data,self.label = self.load_img_file(txt_file,resize,flip,crop,pretrained_model)
    
    def load_img_file(self,txt_file,resize,flip,crop,pre):
        print('------Start loading img files in "%s"------'%(txt_file))
        print('-- 1. Resize img to %s'%(resize,))
        print('-- 2. Crop : %r  , Flip : %r'%(crop,flip))
        t0 = time.time()
        img_list = []
        label_list = []
        file = open(txt_file,'r')
        for line in file:
            img_name,label = line.strip().split(' ')
            img = Image.open(img_name)
            ## Random Flip
            if flip == True:
                img = self.t_flip(img)
            if resize != None:
                if pre == True:
                    tmp_img = self.t_norm(self.to_tensor(self.t_resize(img)))
                else:
                    tmp_img = self.to_tensor(self.t_resize(img))
                img_list.append(tmp_img)
                label_list.append(int(label))
            if crop == True:
                for i in range(self.n_crop):
                    if pre == True:
                        tmp_img = self.t_norm(self.to_tensor(self.t_crop(self.t_b_resize(img))))
                    else:
                        tmp_img = self.to_tensor(self.t_crop(self.t_b_resize(img)))
                    img_list.append(tmp_img)
                    label_list.append(int(label))
        file.close()
        self.n_id = len(set(label_list))

        print('------End loading dataset... |    cost time: %.2f s------'%(time.time()-t0))
        return torch.stack(img_list),torch.LongTensor(label_list)

    def __getitem__(self,idx):
        return {'img':self.data[idx],'gt':self.label[idx]}

    def __len__(self):
        return self.data.size(0)
            


def Get_DataLoader(dataset,batch_size=128,shuffle=True):
    return DataLoader(dataset,batch_size = batch_size,shuffle=True)



if __name__ == '__main__':
    
    pretrained = True

    D = V_Re_ID_Dataset(sys.argv[1],crop=True,flip=True,pretrained_model=pretrained)
    loader = Get_DataLoader(D)

    print('len:',len(D))
    print('n_id:',D.n_id)
    print('n_batch:',len(D)//128+1)

    model = models.ResNet(D.n_id,n_layers=18,pretrained=pretrained).cuda()

    for i,data in enumerate(loader):
        x = data['img']
        y = data['gt']

        output = model(Variable(x).cuda())
        print(output.size())
        break
    



# def generate_txt():
    # path = sys.argv[1]

    # img = glob.glob(path)
    # file = open('train_info.txt','w')
    # for i in range(len(img)):
        # file.write(img[i]+' %d'%(i))
        # file.write('\n')

