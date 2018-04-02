import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import TripletImage_Dataset, sv_comp_Dataset, BoxCars_Dataset, Unsupervised_TripletImage_Dataset, Get_train_DataLoader, Get_val_DataLoader,Get_test_DataLoader
from loss import TripletLoss
from logger import Logger
import models
import sys
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np
cudnn.benchmark=True

def train_joint(args, train_veri_dataloader,
                train_compcars_dataloader, val_compcars_dataloader, 
                train_boxcars_dataloader, val_boxcars_dataloader, 
                train_aic_dataloader, 
                test_compcars_dataloader,
                test_boxcars_dataloader,
                base_net, veri_id_net, color_net, compcars_model_net, boxcars_model_net):

    optimizer_base = optim.Adam(base_net.parameters(), lr=args.lr)
    optimizer_veri = optim.Adam(veri_id_net.parameters(), lr=args.lr)
    optimizer_color = optim.Adam(color_net.parameters(), lr=args.lr)
    optimizer_compcars = optim.Adam(compcars_model_net.parameters(), lr=args.lr)
    optimizer_boxcars = optim.Adam(boxcars_model_net.parameters(), lr=args.lr)
    criterion_triplet = TripletLoss(margin=margin, batch_hard=args.batch_hard)
    criterion_ce = nn.CrossEntropyLoss()
    logger = Logger(os.path.join(args.save_model_dir,'train'))
    val_logger = Logger(os.path.join(args.save_model_dir,'val'))
    test_logger = Logger(os.path.join(args.save_model_dir,'test'))
    
    epoch_size = min(len(train_veri_dataloader), len(train_compcars_dataloader), len(train_boxcars_dataloader), len(train_aic_dataloader))
    for e in range(args.n_epochs):
        
        pbar = tqdm(total=epoch_size,ncols=100,leave=True)
        pbar.set_description('Epoch %d'%(e))
        
        for n in range(epoch_size):
            logger.append_epoch(e + float(n)/epoch_size)
            # VeRi dataset
            epoch_loss = 0
            for i, samples in enumerate(train_veri_dataloader):
                if i==1: break
                imgs = samples['img'].view(samples['img'].size(0)*samples['img'].size(1),
                                           samples['img'].size(2), 
                                           samples['img'].size(3),
                                           samples['img'].size(4))
                classes = samples['class'].view(-1)
                colors = samples['color'].view(-1)
                b_img = Variable(imgs).cuda()
                classes = Variable(classes).cuda()
                colors = Variable(colors).cuda()
                base_net.zero_grad()
                veri_id_net.zero_grad()
                color_net.zero_grad()
                #forward
                pred_feat = base_net(b_img)
                pred_id = veri_id_net(pred_feat)
                pred_color = color_net(pred_feat)
                b_loss_triplet = criterion_triplet(pred_feat, classes)
                loss_id = criterion_ce(pred_id, classes)
                loss_color = criterion_ce(pred_color, colors)
                loss = b_loss_triplet.mean() + loss_id + loss_color
                epoch_loss += loss.data[0]
                # backward
                loss.backward()
                optimizer_base.step()
                optimizer_veri.step()
                optimizer_color.step()
    
                logger.logg({'loss_veri_triplet': b_loss_triplet.data.mean(),
                            'loss_veri_triplet_max': b_loss_triplet.data.max(),
                            'loss_veri_triplet_min': b_loss_triplet.data.min(),
                            'loss_veri_id': loss_id.data[0],
                            'loss_veri_color': loss_color.data[0]})

            # Compcars dataset
            for i, samples in enumerate(train_compcars_dataloader):
                if i==1: break
                img = Variable(samples['img']).cuda()
                model = Variable(samples['model']).cuda()
                color = Variable(samples['color']).cuda()
                base_net.zero_grad()
                compcars_model_net.zero_grad()
                color_net.zero_grad()
                #forward
                pred_feat = base_net(img)
                pred_model = compcars_model_net(pred_feat)
                pred_color = color_net(pred_feat)
                loss_model = criterion_ce(pred_model, model)
                loss_color = criterion_ce(pred_color, color)
                loss = loss_model + loss_color
                epoch_loss += loss.data[0]
                # backward
                loss.backward()
                optimizer_base.step()
                optimizer_compcars.step()
                optimizer_color.step()
    
                logger.logg({'loss_compcars_model': loss_model.data[0],
                            'loss_compcars_color': loss_color.data[0]})

            # Boxcars dataset
            for i, samples in enumerate(train_boxcars_dataloader):
                if i==1: break
                img = Variable(samples['img']).cuda()
                model = Variable(samples['model']).cuda()
                base_net.zero_grad()
                boxcars_model_net.zero_grad()
                #forward
                pred_feat = base_net(img)
                pred_model = boxcars_model_net(pred_feat)
                loss_model = criterion_ce(pred_model, model)
                loss = loss_model
                epoch_loss += loss.data[0]
                # backward
                loss.backward()
                optimizer_base.step()
                optimizer_boxcars.step()
    
                logger.logg({'loss_boxcars_model': loss_model.data[0]})

            # AIC dataset
            for i, samples in enumerate(train_aic_dataloader):
                if i==1: break
                imgs = samples['img'].squeeze(0)
                pos_mask = samples['pos_mask'].squeeze(0)
                neg_mask = samples['neg_mask'].squeeze(0)
                b_img = Variable(imgs).cuda()
                pos_mask = Variable(pos_mask).cuda()
                neg_mask = Variable(neg_mask).cuda()
                base_net.zero_grad()
                #forward
                pred_feat = base_net(b_img)
                b_loss = criterion_triplet(pred_feat, pos_mask=pos_mask, neg_mask=neg_mask, mode='mask')
                loss = b_loss.mean()
                epoch_loss += loss.data[0]
                # backward
                loss.backward()
                optimizer_base.step()
    
                logger.logg({'loss_aic_triplet': b_loss.data.mean(),
                            'loss_aic_triplet_max': b_loss.data.max(),
                            'loss_aic_triplet_min': b_loss.data.min()})

            pbar.update(1)
            pbar.set_postfix({'loss':'%f'%(epoch_loss/(n+1))})
        pbar.close()
        print('Training total loss = %f'%(epoch_loss/epoch_size))
        
        if e % args.save_every_n_epoch == 0:
            torch.save(base_net.state_dict(),os.path.join(args.save_model_dir,'model_%d_base.ckpt'%(e)))
            torch.save(veri_id_net.state_dict(),os.path.join(args.save_model_dir,'model_%d_veri_id.ckpt'%(e)))
            torch.save(compcars_model_net.state_dict(),os.path.join(args.save_model_dir,'model_%d_compcars_model.ckpt'%(e)))
            torch.save(color_net.state_dict(),os.path.join(args.save_model_dir,'model_%d_color.ckpt'%(e)))
            torch.save(boxcars_model_net.state_dict(),os.path.join(args.save_model_dir,'model_%d_boxcars_model.ckpt'%(e)))
        logger.write_log()

        print('start validation')
        val_logger.append_epoch(e)
        base_net.eval()
        veri_id_net.eval()
        color_net.eval()
        compcars_model_net.eval()
        boxcars_model_net.eval()
        '''
        # VeRi
        correct = []
        for i,sample in enumerate(val_veri_dataloader):
            imgs = sample['img'].view(sample['img'].size(0)*sample['img'].size(1),
                                       sample['img'].size(2), 
                                       sample['img'].size(3),
                                       sample['img'].size(4))
            classes = sample['class'].view(sample['class'].size(0)*sample['class'].size(1))
            img = Variable(imgs,volatile=True).cuda()
            gt = Variable(classes,volatile=True).cuda()
            pred = veri_id_net(base_net(img))
            _, pred_cls = torch.max(pred,dim=1)
            correct.append(pred_cls.data==gt.data)
        acc = torch.cat(correct).float().mean()
        print('VeRi ID val acc: %.3f' % acc)
        val_logger.logg({'veri_id_acc':acc})
        '''
        # Compcars
        correct_model = []
        correct_color = []
        for i,sample in enumerate(val_compcars_dataloader):
            img = Variable(sample['img'],volatile=True).cuda()
            gt_model = Variable(sample['model'],volatile=True).cuda()
            gt_color = Variable(sample['color'],volatile=True).cuda()
            pred_feat = base_net(img)
            pred_model = compcars_model_net(pred_feat)
            pred_color = color_net(pred_feat)
            _, pred_model = torch.max(pred_model,dim=1)
            _, pred_color = torch.max(pred_color,dim=1)
            correct_model.append(pred_model.data == gt_model.data)
            correct_color.append(pred_color.data == gt_color.data)
        acc_model = torch.cat(correct_model).float().mean()
        acc_color = torch.cat(correct_color).float().mean()
        print('CompCars model val acc: %.3f' % acc_model)
        print('CompCars color val acc: %.3f' % acc_color)
        val_logger.logg({'compcars_model_acc':acc_model})
        val_logger.logg({'compcars_color_acc':acc_color})
        # Boxcars
        correct_model = []
        for i,sample in enumerate(val_boxcars_dataloader):
            img = Variable(sample['img'],volatile=True).cuda()
            gt_model = Variable(sample['model'],volatile=True).cuda()
            pred_feat = base_net(img)
            pred_model =boxcars_model_net(pred_feat)
            _, pred_model = torch.max(pred_model,dim=1)
            correct_model.append(pred_model.data == gt_model.data)
        acc_model = torch.cat(correct_model).float().mean()
        print('BoxCars model val acc: %.3f' % acc_model)
        val_logger.logg({'boxcars_model_acc':acc_model})
        val_logger.write_log()
        
        if e%25 == 0:
            print('start testing')
            test_logger.append_epoch(e)
            pbar = tqdm(total=len(test_boxcars_dataloader),ncols=100,leave=True)
            pbar.set_description('Test BoxCar')
            correct_model = []
            for i,sample in enumerate(test_boxcars_dataloader):
                img = Variable(sample['img'],volatile=True).cuda()
                gt_model = Variable(sample['model'],volatile=True).cuda()
                pred_feat = base_net(img)
                pred_model =boxcars_model_net(pred_feat)
                _, pred_model = torch.max(pred_model,dim=1)
                correct_model.append(pred_model.data == gt_model.data)
                pbar.update(1)
            pbar.close()
            acc_model = torch.cat(correct_model).float().mean()
            print('BoxCars model val acc: %.3f' % acc_model)
            test_logger.logg({'boxcars_model_acc':acc_model})

            pbar = tqdm(total=len(test_compcars_dataloader),ncols=100,leave=True)
            pbar.set_description('Test CompCar_sv')
            correct_model = []
            correct_color = []
            for i,sample in enumerate(test_compcars_dataloader):
                img = Variable(sample['img'],volatile=True).cuda()
                gt_model = Variable(sample['model'],volatile=True).cuda()
                gt_color = Variable(sample['color'],volatile=True).cuda()
                pred_feat = base_net(img)
                pred_model = compcars_model_net(pred_feat)
                pred_color = color_net(pred_feat)
                _, pred_model = torch.max(pred_model,dim=1)
                _, pred_color = torch.max(pred_color,dim=1)
                correct_model.append(pred_model.data == gt_model.data)
                correct_color.append(pred_color.data == gt_color.data)
                pbar.update(1)
            pbar.close()
            acc_model = torch.cat(correct_model).float().mean()
            acc_color = torch.cat(correct_color).float().mean()
            print('CompCars model val acc: %.3f' % acc_model)
            print('CompCars color val acc: %.3f' % acc_color)
            test_logger.logg({'compcars_model_acc':acc_model})
            test_logger.logg({'compcars_color_acc':acc_color})
            test_logger.write_log()

        base_net.train()
        veri_id_net.train()
        color_net.train()
        compcars_model_net.train()
        boxcars_model_net.train()


if __name__ == '__main__':
    ## Parse arg
    parser = argparse.ArgumentParser(description='Train Re-ID net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--veri_txt', required=True, help='txt for VeRi dataset')
    parser.add_argument('--compcars_txt', required=True, help='txt for CompCars sv dataset')
    parser.add_argument('--compcars_test_txt', required=True, help='txt for CompCars sv dataset')
    parser.add_argument('--boxcars_txt', required=True, help='txt for BoxCars116k dataset')
    parser.add_argument('--boxcars_test_txt', required=True, help='txt for BoxCars116k dataset')
    parser.add_argument('--aic_pkl', required=True, help='pkl for AIC dataset')
    parser.add_argument('--crop',type=bool,default=True,help='Whether crop the images')
    parser.add_argument('--flip',type=bool,default=True,help='Whether randomly flip the image')
    parser.add_argument('--jitter',type=int,default=0,help='Whether randomly jitter the image')
    parser.add_argument('--pretrain',type=bool,default=True,help='Whether use pretrained model')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size number')
    parser.add_argument('--n_epochs',type=int,default=20,help='number of training epochs')
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--save_model_dir',default=None,help='path to save model')
    parser.add_argument('--n_layer',type=int,default=18,help='number of Resnet layers')
    parser.add_argument('--margin',type=str,default='0',help='margin of triplet loss ("soft" or float)')
    parser.add_argument('--class_in_batch',type=int,default=32,help='# of class in a batch for triplet training')
    parser.add_argument('--image_per_class_in_batch',type=int,default=4,help='# of images of each class in a batch for triplet training')
    parser.add_argument('--batch_hard',action='store_true',help='whether to use batch_hard for triplet loss')
    parser.add_argument('--save_every_n_epoch',type=int,default=1,help='save model every n epoch')
    parser.add_argument('--class_w',type=float,help='wieghting of classification loss when triplet training')
    args = parser.parse_args()
    margin = args.margin if args.margin=='soft' else float(args.margin)
    assert args.class_in_batch*args.image_per_class_in_batch == args.batch_size, \
           'batch_size need to equal class_in_batch*image_per_class_in_batch'

    # Load dataset
    #veri_dataset = load_dataset(args.veri_txt)
    #compcars_dataset = load_dataset(args.compcars_txt)
    #boxcar_dataset = load_dataset(args.boxcar_txt)
    #id_dataset = ID_Dataset(veri_dataset)
    #model_dataset = Model_Dataset(compcars_dataset, boxcars_dataset)
    #color_dataset = Color_Dataset(veri_dataset, compcars_dataset)
    #aic_dataset = AIC_Dataset(args.aic_pkl)

    # Get Dataset & DataLoader    
    veri_dataset = TripletImage_Dataset(args.veri_txt, crop=args.crop, flip=args.flip, jitter=args.jitter, 
                                        imagenet_normalize=args.pretrain, val_split=0.01,
                                        class_in_batch=args.class_in_batch,
                                        image_per_class_in_batch=args.image_per_class_in_batch)
    train_veri_dataloader = Get_train_DataLoader(veri_dataset, batch_size=args.class_in_batch)

    compcars_dataset = sv_comp_Dataset(args.compcars_txt, crop=args.crop, flip=args.flip, jitter=args.jitter,
                                       imagenet_normalize=args.pretrain, val_split=0.1)
    train_compcars_dataloader = Get_train_DataLoader(compcars_dataset, batch_size=args.batch_size)
    val_compcars_dataloader = Get_val_DataLoader(compcars_dataset, batch_size=args.batch_size)

    boxcars_dataset = BoxCars_Dataset(args.boxcars_txt, crop=args.crop, flip=args.flip, jitter=args.jitter,
                                       imagenet_normalize=args.pretrain, val_split=0.1) 
    train_boxcars_dataloader = Get_train_DataLoader(boxcars_dataset, batch_size=args.batch_size) 
    val_boxcars_dataloader = Get_val_DataLoader(boxcars_dataset, batch_size=args.batch_size)

    aic_dataset = Unsupervised_TripletImage_Dataset(args.aic_pkl, crop=args.crop, flip=args.flip, jitter=args.jitter, 
                                           imagenet_normalize=args.pretrain, batch_size=args.batch_size)
    train_aic_dataloader = Get_train_DataLoader(aic_dataset, batch_size=1)

    # Test Dataset & loader

    compcars_dataset_test = sv_comp_Dataset(args.compcars_test_txt, crop=False, flip=False, jitter=False,imagenet_normalize=True)
    test_compcars_dataloader = Get_test_DataLoader(compcars_dataset_test, batch_size=args.batch_size)

    boxcars_dataset = BoxCars_Dataset(args.boxcars_test_txt, crop=False, flip=False, jitter=False, imagenet_normalize=True) 
    test_boxcars_dataloader = Get_test_DataLoader(boxcars_dataset, batch_size=args.batch_size) 

    # Get Model
    base_net = models.FeatureResNet(n_layers=args.n_layer, pretrained=args.pretrain)
    veri_id_net = models.NLayersFC(base_net.output_dim, veri_dataset.n_id)
    color_net = models.NLayersFC(base_net.output_dim, 12)
    compcars_model_net = models.NLayersFC(base_net.output_dim, compcars_dataset.n_models)
    boxcars_model_net = models.NLayersFC(base_net.output_dim, boxcars_dataset.n_models)

    if torch.cuda.is_available():
        base_net.cuda()
        veri_id_net.cuda()
        color_net.cuda()
        compcars_model_net.cuda()
        boxcars_model_net.cuda()
    
    if args.save_model_dir !=  None:
        os.system('mkdir -p %s' % os.path.join(args.save_model_dir))

    # Train
    train_joint(args, train_veri_dataloader, \
                train_compcars_dataloader, val_compcars_dataloader, \
                train_boxcars_dataloader, val_boxcars_dataloader, \
                train_aic_dataloader, \
                test_compcars_dataloader,\
                test_boxcars_dataloader,\
                base_net, veri_id_net, color_net, compcars_model_net, boxcars_model_net)

