import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import VReID_Dataset,TripletImage_Dataset,Get_train_DataLoader,Get_val_DataLoader
import models
import sys
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np
cudnn.benchmark=True


def train_ict(args,Dataset,train_Dataloader,val_Dataloader,net):

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
            b_c = Variable(samples['color'].squeeze(1)).cuda()
            b_t = Variable(samples['type'].squeeze(1)).cuda()
            b_size = b_img.size(0)
            net.zero_grad()
            #forward
            b_pred,b_cpred,b_tpred = net(b_img)
            loss = criterion(b_pred,b_gt)+criterion(b_cpred,b_c)+criterion(b_tpred,b_t)
            epoch_loss += loss.data[0]
            # backward
            loss.backward()

            optimizer.step()
            pbar.update(b_size)
            pbar.set_postfix({'loss':'%.2f'%(loss.data[0])})
        pbar.close()
        print('Training total loss = %.3f'%(epoch_loss/iter_count))
        torch.save(net.state_dict(),os.path.join(args.save_model_dir,'model_%d.ckpt'%(e)))
        print('start validation')
        correct_i = []
        correct_c = []
        correct_t = []
        for i,sample in enumerate(val_Dataloader):
            img = Variable(sample['img'],volatile=True).cuda()
            gt = sample['gt']
            c = sample['color']
            t = sample['type']
            net.eval()
            pred,color,type = net(img)
            pred_cls = torch.max(pred.cpu().data,dim=1)[1]
            pred_color = torch.max(color.cpu().data,dim=1)[1]
            pred_type = torch.max(type.cpu().data,dim=1)[1]
            for x in range(gt.size(0)):
                if gt[x][0] == pred_cls[x]:
                    correct_i.append(1)
                else:
                    correct_i.append(0)
                if c[x][0] == pred_color[x]:
                    correct_c.append(1)
                else:
                    correct_c.append(0)
                if t[x][0] == pred_type[x]:
                    correct_t.append(1)
                else:
                    correct_t.append(0)
        print('val acc: id:%.3f, color:%.3f, type:%.3f'%(np.mean(correct_i),np.mean(correct_c),np.mean(correct_t)))
        net.train()
        file = open(os.path.join(args.save_model_dir,'val_log.txt'),'a')
        file.write('Epoch %d: val_id_acc = %.3f, val_color_acc = %.3f, val_type_acc= %.3f\n'%(e,np.mean(correct_i),np.mean(correct_c),np.mean(correct_t)))
        file.close()

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
            b_pred,_ = net(b_img)
            loss = criterion(b_pred,b_gt)
            epoch_loss += loss.data[0]
            # backward
            loss.backward()

            optimizer.step()
            pbar.update(b_size)
            pbar.set_postfix({'loss':'%.2f'%(loss.data[0])})
        pbar.close()
        print('Training total loss = %.3f'%(epoch_loss/iter_count))
        torch.save(net.state_dict(),os.path.join(args.save_model_dir,'model_%d.ckpt'%(e)))
        print('start validation')
        net.eval()
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
        net.train()
        file = open(os.path.join(args.save_model_dir,'val_log.txt'),'a')
        file.write('Epoch %d: val_acc = %.3f\n'%(e,np.mean(correct)))
        file.close()

def train_triplet(args,Dataset,train_Dataloader,val_Dataloader,net):

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion_triplet = nn.MarginRankingLoss(margin=args.margin)
    if args.with_class: criterion_class = nn.CrossEntropyLoss()
    
    for e in range(args.n_epochs):
        
        pbar = tqdm(total=len(Dataset.train_index),ncols=100,leave=True)
        pbar.set_description('Epoch %d'%(e))

        epoch_loss = 0
        iter_count = 0
        for i_batch,samples in enumerate(train_Dataloader):
            iter_count +=1
            b_img1 = Variable(samples['img1']).cuda()
            b_img2 = Variable(samples['img2']).cuda()
            b_img3 = Variable(samples['img3']).cuda()
            if args.with_class:
                b_class1 = Variable(samples['class1']).cuda()
                b_class2 = Variable(samples['class2']).cuda()
                b_class3 = Variable(samples['class3']).cuda()
            b_size = b_img1.size(0)
            target = Variable(-1*torch.ones(b_size, 1)).cuda()
            net.zero_grad()
            #forward
            dist_pos, dist_neg, b_pred1, b_pred2, b_pred3 = net(b_img1, b_img2, b_img3)
            loss = criterion_triplet(dist_pos, dist_neg, target)
            if args.with_class:
                loss += criterion_class(b_pred1, b_class1)
                loss += criterion_class(b_pred2, b_class2)
                loss += criterion_class(b_pred3, b_class3)
            epoch_loss += loss.data[0]
            # backward
            loss.backward()

            optimizer.step()
            pbar.update(b_size)
            pbar.set_postfix({'loss':'%.2f'%(loss.data[0])})
        pbar.close()
        print('Training total loss = %.3f'%(epoch_loss/iter_count))
        torch.save(net.net.state_dict(),os.path.join(args.save_model_dir,'model_%d.ckpt'%(e)))
        print('start validation')
        net.eval()
        correct = []
        for i,sample in enumerate(val_Dataloader):
            img1 = Variable(sample['img1'],volatile=True).cuda()
            img2 = Variable(sample['img2'],volatile=True).cuda()
            img3 = Variable(sample['img3'],volatile=True).cuda()
            dist_pos, dist_neg, _, _, _ = net(img1,img2,img3)
            dist_pos, dist_neg = dist_pos.data, dist_neg.data
            for x in range(dist_pos.size(0)):
                if dist_pos[x, 0] < dist_neg[x, 0]:
                    correct.append(1)
                else:
                    correct.append(0)
        print('val acc: %.3f'%(np.mean(correct)))
        net.train()
        file = open(os.path.join(args.save_model_dir,'val_log.txt'),'a')
        file.write('Epoch %d: val_acc = %.3f\n'%(e,np.mean(correct)))
        file.close()

if __name__ == '__main__':
    ## Parse arg
    parser = argparse.ArgumentParser(description='Train Re-ID net',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--info',default='train_info.txt',help='txt file contain path of data')
    parser.add_argument('--crop',type=bool,default=True,help='Whether crop the images')
    parser.add_argument('--flip',type=bool,default=True,help='Whether randomly flip the image')
    parser.add_argument('--jitter',type=int,default=0,help='Whether randomly jitter the image')
    parser.add_argument('--pretrain',type=bool,default=True,help='Whether use pretrained model')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size number')
    parser.add_argument('--n_epochs',type=int,default=20,help='number of training epochs')
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--save_model_dir',default=None,help='path to save model')
    parser.add_argument('--n_layer',type=int,default=18,help='number of Resnet layers')
    parser.add_argument('--dataset',default='VeRi',help='which dataset:VeRi or VeRi_ict')
    parser.add_argument('--triplet',default=None,help='white-space seperated txt containing triplet sample in each row, with column order (anchor, positve, negative)')
    parser.add_argument('--with_class',action='store_true', help='whether to train with class label during triplet training')
    parser.add_argument('--margin',type=int,default=0,help='margin of triplet loss')
    args = parser.parse_args()

    ## Get Dataset & DataLoader
    if args.triplet:
        Dataset = TripletImage_Dataset(args.info, args.triplet, crop=args.crop, flip=args.flip, jitter=args.jitter, imagenet_normalize=args.pretrain, classification=args.with_class)
    else:
        Dataset = VReID_Dataset(args.info,crop=args.crop,flip=args.flip,jitter=args.jitter,pretrained_model=args.pretrain,dataset=args.dataset)
    train_Dataloader = Get_train_DataLoader(Dataset,batch_size=args.batch_size)
    val_Dataloader = Get_val_DataLoader(Dataset,batch_size=args.batch_size)

    ## Get Model
    if args.triplet:
        net = models.TripletNet(models.ResNet(Dataset.n_id,n_layers=args.n_layer,pretrained=args.pretrain))
    else:
        if args.dataset != 'VeRi_ict':
            net = models.ResNet(Dataset.n_id,n_layers=args.n_layer,pretrained=args.pretrain)
        else:
            net = models.ICT_ResNet(Dataset.n_id,Dataset.n_color,Dataset.n_type,n_layers=args.n_layer,pretrained=args.pretrain)
    if torch.cuda.is_available():
        net.cuda()
    
    if args.save_model_dir !=  None:
        os.system('mkdir -p %s' % os.path.join(args.save_model_dir))
    ## train
    if args.load_ckpt == None:
        print('total data:',len(Dataset))
        print('training data:',Dataset.n_train)
        print('validation data:',Dataset.n_val)
        if args.triplet:
            train_triplet(args, Dataset, train_Dataloader, val_Dataloader, net)
        else:
            if args.dataset != 'VeRi_ict':
                train(args,Dataset,train_Dataloader,val_Dataloader,net)
            else:
                train_ict(args,Dataset,train_Dataloader,val_Dataloader,net)






