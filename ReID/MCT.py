from ReID_CNN.Model_Wrapper import ResNet_Loader 
from track import Track

import argparse
import numpy as np
import os
from scipy.misc import imsave,imread
from progressbar import ProgressBar
from tqdm import tqdm
import sys
import csv
import torch
import math
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

def dump_img(self,opt,class_):
    output_dir = os.path.join(opt.cluster_method,'%05d'%(class_))
    os.system('mkdir -p %s' % output_dir)
    input_list = []
    for i in range(len(self.history)):
        input_list += [self.history[i]['loc']+'/%05dF%05d.jpg'%(self.history[i]['id'],frame_number) for frame_number in self.history[i]['frames']]
    output_list = []
    for img_path in input_list:
        p = imread(os.path.join(opt.img_dir,img_path))
        p_path = os.path.join(output_dir,img_path.replace('/','_'))
        output_list.append((p,p_path))
    for tup in output_list:
        imsave(tup[1],tup[0])
def dump_log(self,file,c):
    file.write('Cluster Class %05d :\n'%(c))
    for info in self.history:
        file.write('%s_id : %05d, Loc : %s\n'%(info['attribute'],info['id'],info['loc']))
    file.write('-'*50)
    file.write('\n')

def multi_camera_matching(opt,MCT):
    if opt.cluster_method == 'k-means':
        from sklearn.cluster import KMeans
        # decide n_class to be clustered 
        n_tracker = sum([len(loc_tracker) for loc_tracker in MCT])
        n_classes = n_tracker - 100 * (len(MCT)-1)
        
        all_features = []
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                all_features.append(MCT[i][j].summarized_feature())
        all_features = np.stack(all_features,axis=0)

        print('start kmeans')
        Cluster = KMeans(n_clusters=n_classes,precompute_distances=True,n_jobs=-1)
        class_output = Cluster.fit_predict(all_features)
        print('end kmeans')
        
        # dict order by clustered class
        classified_Trackers = defaultdict(list)
        count = 0
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                class_ = class_output[count]
                classified_Trackers[class_].append(MCT[i][j])
                count += 1
        
        ##Check every cluster class 
        for c in classified_Trackers.keys():
            Trackers = classified_Trackers[c]
            # maybe need to sort by start frames
            # Trackers.sort(key=lambda x:x.get_head_frame())
            if len(Trackers) > 1:
                for t in range(len(Trackers)-1):
                    Trackers[-1].mct_merge(Trackers[t])
                classified_Trackers[c] = Trackers[-1]
            else:
                classified_Trackers[c] = Trackers[-1]

        #visualize
        #for c,tracker in classified_Trackers.items():
        #    tracker.dump_img(opt,c)
        # dump log 
        #file = open(opt.log_txt,'w')
        #for c,tracker in classified_Trackers.items():
        #    tracker.dump_log(file,c)
        #file.close()
        
        return MCT # TODO: should only output tracks that travel all 4 locations

if __name__ == '__main__':
    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('tracks_dir', help='tracks obj pickle directory')
    parser.add_argument('--cluster_method', default='k-means', type=str, help='cluster methods')
    #parser.add_argument('--log_txt', default='./reid_log.txt', type=str, help='Final tracker log')

    args = parser.parse_args()

    # Create sequence names
    loc = [1]#[1, 2, 3, 4]
    loc_n_seq = [1]#[4, 6, 2, 3]
    loc_list = []
    for i, l in enumerate(loc):
        for n in range(1,loc_n_seq[i]+1):
            loc_list.append('Loc%d_%d' % (l, n))
    
    # Load and initialize tracks objs with seq names
    multi_cam_tracks = []
    i = 0
    for i, l in enumerate(loc):
        single_cam_tracks = []
        for n in range(1,loc_n_seq[i]+1):
            name = 'Loc%d_%d' % (l, n)
            pkl_name = os.path.join(args.tracks_dir, '%s.pkl' % name)
            with open(pkl_name, 'rb') as f:
                tracks = pickle.load(f)
            for t in tracks:
                t.assign_seq_id(i)
            single_cam_tracks += tracks
        multi_cam_tracks.append(single_cam_tracks)

    # Multi camera matching
    tracks = multi_camera_matching(args, multi_cam_tracks)

    # Output to file
    dets = []
    for t0 in multi_cam_tracks:
        for t1 in t0:
            dets.append(t1.dump())
    dets = np.concatenate(dets, axis=0)
    dets = np.concatenate([dets[:,:7], -1*np.ones((dets.shape[0],1)), dets[:,[7]]], axis=1)
    np.savetxt('track3.txt', dets, fmt='%f')
