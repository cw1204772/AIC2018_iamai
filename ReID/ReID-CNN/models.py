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
            if name =='fc':
                x = x.view(x.size(0),-1)
            x = module(x)
        return x

if __name__ == '__main__':
    netM = ResNet(n_id=1000,n_layers=18,pretrained=True).cuda()

    print(netM)
    output = netM(Variable(torch.ones(3,224,224).cuda()/2.))
    print(output.size())
