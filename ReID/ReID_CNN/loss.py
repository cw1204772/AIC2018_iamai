import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class TripletLoss(nn.Module):
    '''
    Original margin ranking loss:
        loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    
    Let z = -y * (x1 - x2)

    Soft_margin mode:
        loss(x1, x2, y) = log(1 + exp(z))
    Batch_hard mode:
        z = -y * (x1' - x2'),
        where x1' is the max x1 within a batch,
        x2' is the min x2 within a batch
    '''
    def __init__(self, margin=0, batch_hard=False):
        """
        Args:
            margin: int or 'soft'
            batch_hard: whether to use batch_hard loss
        """
        super(TripletLoss, self).__init__()
        self.batch_hard = batch_hard
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id'):
        dist = self.cdist(feat, feat)
        if mode == 'id':
            if id is None:
                 raise RuntimeError('foward is in id mode, please input id!')
            else:
                 identity_mask = Variable(torch.eye(feat.size(0)).byte())
                 identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                 same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                 negative_mask = same_id_mask ^ 1
                 positive_mask = same_id_mask ^ identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                 raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                 positive_mask = pos_mask
                 same_id_mask = neg_mask ^ 1
        else:
            raise ValueError('unrecognized mode')
        if self.batch_hard:
            max_positive = (dist * positive_mask.float()).max(1)[0]
            min_negative = (dist + 1e5*same_id_mask.float()).min(1)[0]
            z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1,1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1,1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative
        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")
        return b_loss
            
    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b
        
        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff**2).sum(2)+1e-12).sqrt()

if __name__ == '__main__':
    criterion0 = TripletLoss(margin=0.5, batch_hard=False)
    criterion1 = TripletLoss(margin=0.5, batch_hard=True)
    
    t = np.random.randint(3, size=(10,))
    print(t)
    
    feat = Variable(torch.rand(10, 2048), requires_grad=True).cuda()
    id = Variable(torch.from_numpy(t), requires_grad=True).cuda()
    loss0 = criterion0(feat, id)
    loss1 = criterion1(feat, id)
    print('no batch hard:', loss0)
    print('batch hard:', loss1)
    loss0.backward()
    loss1.backward()
