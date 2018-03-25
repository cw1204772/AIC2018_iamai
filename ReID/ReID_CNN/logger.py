import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from collections import OrderedDict

class Logger:
     def __init__(self, save_dir, prefix=''):
         #names = ['epoch',
         #         'loss', 'loss_max', 'loss_median', 'loss_min', 'active_loss',
         #         'feat_2-norm_max', 'feat_2-norm_median', 'feat_2-norm_min']
         self.log = OrderedDict([('epoch', [])])
         self.save_dir = os.path.join(save_dir)
         pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
         self.prefix = prefix

     def logg(self, d):
         for k in d:
             if k not in self.log:
                 self.log[k] = []
             self.log[k].append(d[k])

     def append_epoch(self, e):
         self.log['epoch'].append(e)

     def append_loss(self, b_loss):
         names = ['loss', 'loss_max', 'loss_median', 'loss_min', 'active_loss']
         for n in names:
             if n not in self.log: self.log[n] = []
         self.log['loss'].append(b_loss.mean())
         self.log['loss_max'].append(b_loss.max())
         self.log['loss_median'].append(np.median(b_loss))
         self.log['loss_min'].append(b_loss.min())
         self.log['active_loss'].append((b_loss > 1e-3).mean())
     
     def append_feat(self, b_feat):
         names = ['feat_2-norm_max', 'feat_2-norm_median', 'feat_2-norm_min']
         for n in names:
             if n not in self.log: self.log[n] = []
         norm = np.linalg.norm(b_feat, axis=1)
         self.log['feat_2-norm_max'].append(norm.max())
         self.log['feat_2-norm_median'].append(np.median(norm))
         self.log['feat_2-norm_min'].append(norm.min())

     def write_log(self):
         dataframe = pd.DataFrame(self.log)
         dataframe.to_csv(os.path.join(self.save_dir, '%slog.csv' % self.prefix), index=False)

     def plot(self):
         epoch = np.array(self.log['epoch'])

         plt.figure()
         labels = ['loss_max', 'loss_median', 'loss_min']
         for i, l in enumerate(labels):
             data = np.array(self.log[l])
             plt.semilogy(epoch, data, label=l, color=cm.Blues(0.25+float(i)*0.25))
         data = np.array(self.log['loss'])
         plt.semilogy(epoch, data, label='loss', color='r')
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('loss') 
         plt.title('loss vs. epoch')
         plt.savefig(os.path.join(self.save_dir, '%sloss.png' % self.prefix))
         plt.close()

         plt.figure()
         data = np.array(self.log['active_loss'])
         plt.plot(epoch, data, label='active_loss')
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('% of active loss') 
         plt.title('% of active loss vs. epoch')
         plt.savefig(os.path.join(self.save_dir, '%sactive_loss.png' % self.prefix))
         plt.close()

         plt.figure()
         labels = ['feat_2-norm_max', 'feat_2-norm_median', 'feat_2-norm_min']
         for i, l in enumerate(labels):
             data = np.array(self.log[l])
             plt.plot(epoch, data, label=l, color=cm.Blues(0.25+float(i)*0.25))
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('2-norm of feature') 
         plt.title('2-norm of feature vs. epoch')
         plt.savefig(os.path.join(self.save_dir, '%sfeature_norm.png' % self.prefix))
         plt.close()        





