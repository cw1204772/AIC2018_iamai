import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Logger:
     def __init__(self, save_dir):
         self.log = {'loss':[],
                     'loss_max':[],
                     'loss_median':[],
                     'loss_min':[],
                     'active_loss':[],
                     'feat_2-norm_max':[],
                     'feat_2-norm_median':[],
                     'feat_2-norm_min':[]}
         self.save_dir = save_dir
         pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

     def log(self, stat):
         '''
         Args:
             stat: dict of 2-column list, 
                   1st column is epoch number, 2nd column is stat value
         '''
         for key in stat:
             if key not in self.log: self.log[key] = []
             self.log[key].append(stat[key])

     def log_loss(self, n, b_loss):
         self.log['loss'].append([n, b_loss.mean()])
         self.log['loss_max'].append([n, b_loss.max()])
         self.log['loss_median'].append([n, np.median(b_loss)])
         self.log['loss_min'].append([n, b_loss.min()])
         self.log['active_loss'].append([n, (b_loss > 1e-3).mean()])
     
     def log_feat(self, n, b_feat):
         norm = np.linalg.norm(b_feat, axis=1)
         self.log['feat_2-norm_max'].append([n, norm.max()])
         self.log['feat_2-norm_median'].append([n, np.median(norm)])
         self.log['feat_2-norm_min'].append([n, norm.min()])

     def plot(self):
         plt.figure()
         labels = ['loss_max', 'loss_median', 'loss_min']
         for i, l in enumerate(labels):
             data = np.array(self.log[l])
             plt.plot(data[:,0], data[:,1], label=l, color=cm.Blues(float(i)/len(labels)))
         data = np.array(self.log['loss'])
         plt.plot(data[:,0], data[:,1], label='loss', color='r')
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('loss') 
         plt.title('loss vs. epoch')
         plt.savefig(os.path.join(self.save_dir, 'loss.png'))
         plt.close()

         plt.figure()
         data = np.array(self.log['active_loss'])
         plt.plot(data[:,0], data[:,1], label='active_loss')
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('% of active loss') 
         plt.title('% of active loss vs. epoch')
         plt.savefig(os.path.join(self.save_dir, 'active_loss.png'))
         plt.close()

         plt.figure()
         labels = ['feat_2-norm_max', 'feat_2-norm_median', 'feat_2-norm_min']
         for i, l in enumerate(labels):
             data = np.array(self.log[l])
             plt.plot(data[:,0], data[:,1], label=l, color=cm.Blues(float(i)/len(labels)))
         plt.legend()
         plt.xlabel('epoch') 
         plt.ylabel('2-norm of feature') 
         plt.title('2-norm of feature vs. epoch')
         plt.savefig(os.path.join(self.save_dir, 'feature_norm.png'))
         plt.close()        





