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
        ######################
        self.img_list = []
        self.label_list = []
        file = open(txt_file,'r')
        for row in file:
            img_name,label = row.strip().split(' ')
            self.img_list.append(img_name)
            self.label_list.append(int(label)-1)
        file.close()
        self.n_id = len(set(self.label_list))

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
        return {'img':img,'gt':torch.LongTensor([label])}

    def __len__(self):
        return len(self.img_list)
            


def Get_DataLoader(dataset,batch_size=128,shuffle=True,num_workers=6):
    return DataLoader(dataset,batch_size = batch_size,shuffle=True,num_workers=6)



# def generate_txt():
    # path = sys.argv[1]

    # img = glob.glob(path)
    # print(img)
    # # exit(-1)
    # file = open('train_info.txt','w')
    # for i in range(len(img)):
        # file.write(img[i]+' %d'%(i))
        # file.write('\n')
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
    




