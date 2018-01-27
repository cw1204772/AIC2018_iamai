import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms


class ResNet(nn.Module):
    def __init__(self,n_id,n_layers=50,pretrained=True):
        super(ResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        for name,modules in old_model._modules.items():
            self.add_module(name,modules)
        self.fc = nn.Linear(self.fc.in_features,n_id)
        #########
        self.pretrained = pretrained
    def forward(self,x):
        for name,module in self._modules.items():
            if name != 'fc':
                x = module(x)
        x = self.fc(x.view(x.size(0),-1))
        return x

class ICT_ResNet(nn.Module):
    def __init__(self,n_id,n_color,n_type,n_layers=50,pretrained=True):
        super(ICT_ResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        for name,modules in old_model._modules.items():
            self.add_module(name,modules)
        self.fc = nn.Linear(self.fc.in_features,n_id)
        self.fc_c = nn.Linear(self.fc.in_features,n_color)
        self.fc_t = nn.Linear(self.fc.in_features,n_type)
        #########
        self.pretrained = pretrained
    def forward(self,x):
        for name,module in self._modules.items():
            if name.find('fc')==-1:
                x = module(x)
        x = x.view(x.size(0),-1)
        x_i = self.fc(x)
        x_c = self.fc_c(x)
        x_t = self.fc_t(x)
        return x_i,x_c,x_t
if __name__ == '__main__':
    netM = ICT_ResNet(n_id=1000,n_color=7,n_type=7,n_layers=18,pretrained=True).cuda()

    print(netM)
    output = netM(Variable(torch.ones(1,3,224,224).cuda()/2.))
    print(output[0].size())
    print(output[1].size())
    print(output[2].size())
