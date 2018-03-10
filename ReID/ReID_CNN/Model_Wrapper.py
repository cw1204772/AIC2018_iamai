import torch
import os
import sys
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from progressbar import Bar, ETA, ReverseBar, Percentage, ProgressBar, Timer
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from ReID_CNN.models import ICT_ResNet

class Feature_ResNet(nn.Module):
    def __init__(self,n_layer,output_color):
        super(Feature_ResNet,self).__init__()
        all_model = ICT_ResNet(1,10,9,n_layer,pretrained=False)
        for name,modules in all_model._modules.items():
            if name.find('fc') == -1 :
                self.add_module(name,modules)
        if output_color == True:
            self.fc_c = all_model.fc_c
        self.output_color = output_color
    def forward(self,x):
        for name,module in self._modules.items():
            if name.find('fc') == -1:
                x = module(x)
        x = x.view(x.size(0),-1)
        if self.output_color == False:  return x
        else:
            output  = self.fc_c(x)
            color = torch.max(self.fc_c(x),dim=1)[1]
            return x,color

class ResNet_Loader(object):
    def __init__(self,model_path,n_layer,batch_size=128,output_color=False):
        self.batch_size = batch_size
        self.output_color = output_color

        self.model = Feature_ResNet(n_layer,output_color)
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            if key.find('fc') != -1 and key.find('fc_c') == -1 :
                del state_dict[key]
            elif output_color == False and key.find('fc_c') != -1:
                del state_dict[key]
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('loading resnet%d model'%(n_layer))
        self.compose = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    def inference(self,file_name_list):
         
        self.model.cuda()
        feature_list = []
        color_list = []
        batch_list = []
        widgets = [Timer(format='ET: %(elapsed)s'), Bar('>'), ' ', 'Inferencing: ', Percentage(), ' ', ReverseBar('<'), ETA()]
        pbar = ProgressBar(widgets=widgets, max_value=len(file_name_list))
        for i,name in enumerate(file_name_list):
            img = Image.open(name)
            img = self.compose(img)
            batch_list.append(img)
            if (i+1)% self.batch_size == 0:
                if self.output_color == False:
                    features = self.model(Variable(torch.stack(batch_list)).cuda())
                    feature_list.append(features.cpu().data)
                else:
                    features,colors = self.model(Variable(torch.stack(batch_list)).cuda())
                    feature_list.append(features.cpu().data)
                    color_list.append(colors.cpu().data)
                batch_list = []
                pbar.update(i)
        pbar.finish()
        if len(batch_list)>0:
            if self.output_color == False:
                features = self.model(Variable(torch.stack(batch_list)).cuda())
                feature_list.append(features.cpu().data)
            else:
                features,colors = self.model(Variable(torch.stack(batch_list)).cuda())
                feature_list.append(features.cpu().data)
                color_list.append(colors.cpu().data)
            batch_list = []
        self.model.cpu()
        if self.output_color == False:
            feature_list = torch.cat(feature_list,dim=0)
            return feature_list
        else:
            feature_list = torch.cat(feature_list,dim=0)
            color_list = torch.cat(color_list,dim=0)
            return feature_list,color_list

