import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import V_Re_ID_Dataset,Get_train_DataLoader,Get_val_DataLoader
import models
import sys
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np
cudnn.benchmark=True


def train(args,Dataset,train_Dataloader,val_Dataloader,net):

    optimizer = optim.Adam(net.parameters(),lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.n_epochs):
        
        pbar = tqdm(total=len(Dataset.train_index),ncols=100,leave=True)
        pbar.set_description('Epoch %d'%(e))

        epoch_loss = 0
        iter_count = 0
        for i_batch,samples in enumerate(train_Dataloader):
            iter_count +=1
            b_img = Variable(samples['img']).cuda()
            b_gt = Variable(samples['gt'].squeeze(1)).cuda()
            b_size = b_img.size(0)
            net.zero_grad()
            #forward
            b_pred = net(b_img)
            loss = criterion(b_pred,b_gt)
            epoch_loss += loss.data[0]
            # backward
            loss.backward()

            optimizer.step()
            pbar.update(b_size)
            pbar.set_postfix({'loss':'%.2f'%(loss.data[0])})
        pbar.close()
        print('Training total loss = %.3f'%(epoch_loss/iter_count))
        torch.save(net.state_dict(),os.path.join('ckpt',args.save_model_dir,'model_%d.ckpt'%(e)))
        print('start validation')
        correct = []
        for i,sample in enumerate(val_Dataloader):
            img = Variable(sample['img'],volatile=True).cuda()
            gt = sample['gt']
            pred = net(img)
            pred_cls = torch.max(pred.cpu().data,dim=1)[1]
            for x in range(gt.size(0)):
                if gt[x][0] == pred_cls[x]:
                    correct.append(1)
                else:
                    correct.append(0)
        print('val acc: %.3f'%(np.mean(correct)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Re-ID net')
    parser.add_argument('--info',default='train_info.txt',help='txt file contain path of data')
    parser.add_argument('--crop',type=bool,default=True,help='Whether crop the images')
    parser.add_argument('--flip',type=bool,default=True,help='Whether randomly flip the image')
    parser.add_argument('--pretrain',type=bool,default=True,help='Whether use pretrained model')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=1,help='batch size number')
    parser.add_argument('--n_epochs',type=int,default=50,help='number of training epochs')
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--save_model_dir',default=None,help='path to save model')
    parser.add_argument('--n_layer',type=int,default=50,help='number of Resnet layers')

    args = parser.parse_args()

    ## Get Dataset & DataLoader
    Dataset = V_Re_ID_Dataset(args.info,crop=args.crop,flip=args.flip,pretrained_model=args.pretrain)
    train_Dataloader = Get_train_DataLoader(Dataset,batch_size=args.batch_size)
    val_Dataloader = Get_val_DataLoader(Dataset,batch_size=args.batch_size)

    ## get Model
    net = models.ResNet(Dataset.n_id,n_layers=args.n_layer,pretrained=args.pretrain)
    if torch.cuda.is_available():
        net.cuda()
    
    if args.save_model_dir !=  None:
        os.system('mkdir -p %s' % os.path.join('ckpt', args.save_model_dir))
    ## train
    if args.load_ckpt == None:
        print('total data:',len(Dataset))
        train(args,Dataset,train_Dataloader,val_Dataloader,net)






