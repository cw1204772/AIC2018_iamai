import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import argparse
from collections import defaultdict,OrderedDict
import random
from PIL import Image
from Model_Wrapper import ResNet_Loader
import os
import glob


def compute_cmc(features,labels):
    error = []
    norm_features = nn.functional.normalize(features,dim=1).cuda()
    norm_features_2 = norm_features.transpose(0,1).cuda()
    SimMat = -1*torch.mm(norm_features,norm_features_2).cpu()
    del norm_features,norm_features_2
    
    cmc = torch.zeros(SimMat.size(0))
    for i in range(SimMat.size(0)):
        _,argsort = torch.sort(SimMat[i])
        
        for j in range(SimMat.size(0)):
            if labels[argsort[j]] != labels[i] and argsort[j]!=i and j==1:
                error.append(argsort[j])
            if labels[argsort[j]] == labels[i] and argsort[j]!=i:
                rank = j-1
                break
        for j in range(rank,SimMat.size(0)):
            cmc[j]+=1

    cmc = torch.floor((cmc/SimMat.size(0))*100)  
    return cmc,error



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Re-ID net',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--n_layer',type=int,default=18,help='number of Resnet layers')
    parser.add_argument('--gallery_txt',default=None,help='path to load gallery')
    parser.add_argument('--query_txt',default=None,help='path to load query')
    parser.add_argument('--dis_mat',default=None,help='path to store distance')
    parser.add_argument('--batch_size',default=64,help='batch_size for inferencing')

    args = parser.parse_args()

    print('loading model....')
    model = ResNet_Loader(args.load_ckpt,args.n_layer,output_color=False,batch_size=64)
    
    with open(args.query_txt,'r') as f:
        query_txt = [q.strip() for q in f.readlines()]
        query_txt = query_txt[1:]
    with open(args.gallery_txt,'r') as f:
        gallery_txt = [q.strip() for q in f.readlines()]
        gallery_txt = gallery_txt[1:]
    print('inferencing q_features')
    q_features = model.inference(query_txt)
    print('inferencing g_features')
    g_features = model.inference(gallery_txt)
    
    q_features = nn.functional.normalize(q_features,dim=1).cuda()
    g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()
    
    print('compute distance')
    SimMat = -1 * torch.mm(q_features,g_features)
    SimMat = SimMat.cpu().transpose(0,1)

    print(SimMat.size())
    
    SimMat = SimMat.numpy()
    import scipy.io as sio
    sio.savemat(args.dis_mat,{'dist_CNN':SimMat})
    
