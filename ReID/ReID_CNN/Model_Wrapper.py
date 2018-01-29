import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import models
from torchvision import transforms
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

class ResNet_Loader(object):
    def __init__(self,model_path,n_layer):
        self.model = Feature_ResNet(n_layer)
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            if key.find('fc') != -1:
                del state_dict[key]
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('loading resnet%d model'%(n_layer))
        self.compose = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    def inference(self,file_name_list):
         
        self.model.cuda()
        feature_list = []
        batch_list = []
        for i,name in enumerate(file_name_list):
            img = Image.open(name)
            img = self.compose(img)
            batch_list.append(img)
            if (i+1)% 128 == 0:
                features = self.model(Variable(torch.stack(batch_list)).cuda())
                feature_list.append(features.cpu().data)
                batch_list = []
        if len(batch_list)>0:
            features = self.model(Variable(torch.stack(batch_list)).cuda())
            feature_list.append(features.cpu().data)
            batch_list = []
        feature_list = torch.cat(feature_list,dim=0)
        self.model.cpu()
        return feature_list

