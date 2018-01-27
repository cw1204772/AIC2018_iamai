import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import models
import argparse
from collections import defaultdict,OrderedDict
import random
from PIL import Image


class Feature_ResNet(nn.Module):
    def __init__(self,n_layer):
        super(Feature_ResNet,self).__init__()
        all_model = models.ResNet(1,n_layer,pretrained=False)
        for name,modules in all_model._modules.items():
            if name != 'fc':
                self.add_module(name,modules)
    def forward(self,x):
        for name,module in self._modules.items():
            x = module(x)
        return x.view(x.size(0),-1)
        
def load_model(n_layer,load_ckpt):
    net = Feature_ResNet(n_layer)
    state_dict = torch.load(load_ckpt)
    for key in list(state_dict.keys()):
        if key.find('fc') != -1:
            del state_dict[key]
    net.load_state_dict(state_dict)

    return net

def get_sample_info(info_txt,n_sample):
    info_dict = defaultdict(list)
    file = open(info_txt,'r')
    for row in file:
        img_file,label = row.strip().split(' ')
        label = int(label)-1
        info_dict[label].append(img_file)
    info_dict = OrderedDict(sorted(info_dict.items()))

    g_n = 0
    for k,v in info_dict.items():
        if len(v) > n_sample:
            random.shuffle(v)
            info_dict[k] = v[:n_sample]
        g_n += len(info_dict[k])
    print('sample from test dataset, %d for one ID'%(n_sample))
    return info_dict

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
    parser.add_argument('--info',default='train_info.txt',help='txt file contain path of data')
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--n_layer',type=int,default=50,help='number of Resnet layers')

    args = parser.parse_args()

    # features = torch.load('features')
    # labels = torch.load('labels')
    print('loading model....')
    net = load_model(args.n_layer,args.load_ckpt)
    net.cuda(1)
    net.eval()
    print('get gallery....')
    gallery_dict = get_sample_info(args.info,6)
    compose = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],
                                                                                                          std=[0.229,0.224,0.225])])
    print('get feature....')
    features = []
    labels = []
    for k in gallery_dict.keys():
        ID_img = []
        for name in gallery_dict[k]:
            img = Image.open(name)
            img = compose(img)
            ID_img.append(img)
        ID_img = torch.stack(ID_img)
        output = net(Variable(ID_img).cuda(1))
        features.append(output.cpu().data)
        labels.append(torch.ones(output.size(0)).long()*k)
    # features size :(9383,512) labels size : (9383)
    features = torch.cat(features,dim=0)
    labels = torch.cat(labels)
    
    torch.save(features,'features')
    torch.save(labels,'labels')
    # exit(-1)
    print('compute cmc....')
    cmc,error = compute_cmc(features,labels)
    print(cmc[:100])
    # print(error[:10])
    # img_list = []
    # for values in gallery_dict.values():
        # img_list+=values
    # for i in range(len(error)):
        # print(img_list[i],img_list[error[i]])

    
    
    

