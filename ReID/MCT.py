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
import pathlib
import shutil

def dump_imgs(output_dir, track):
    """Dump images from img_paths to output_dir/id"""
    output_dir = os.path.join(output_dir, '%06d' % track.id)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for path in track.dump_img_paths():
        p = path.split('/')
        name = p[-2] + '_' + p[-1]
        shutil.copyfile(path, os.path.join(output_dir, name))

def dump_log(self,file,c):
    file.write('Cluster Class %05d :\n'%(c))
    for info in self.history:
        file.write('%s_id : %05d, Loc : %s\n'%(info['attribute'],info['id'],info['loc']))
    file.write('-'*50)
    file.write('\n')

def multi_camera_matching(opt,MCT):
    print('multi camera matching...')
    if opt.cluster_method == 'k-means':
        from sklearn.cluster import MiniBatchKMeans
        # decide n_class to be clustered 
        n_tracker = sum([len(loc_tracker) for loc_tracker in MCT])
        #n_classes = n_tracker - 100 * (len(MCT)-1)
        # haven't decide
        n_classes = 1000
        print('clustering %d tracks with %d clusters...' % (n_tracker, n_classes))
        
        all_features = []
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                all_features.append(MCT[i][j].summarized_feature())
        all_features = np.stack(all_features,axis=0)

        print('start kmeans')
        #Cluster = KMeans(n_clusters=n_classes,precompute_distances=True,n_jobs=-1,max_iter=500,n_init=15,verbose=1)
        Cluster = MiniBatchKMeans(n_clusters=n_classes,max_iter=500,n_init=15,init_size=n_classes,verbose=1,max_no_improvement=100)
        class_output = Cluster.fit_predict(all_features)
        print('end kmeans')
        del all_features
        # dict order by clustered class
        classified_Trackers = defaultdict(list)
        count = 0
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                class_ = class_output[count]
                classified_Trackers[class_].append(MCT[i][j])
                count += 1
        
        ##Check every cluster class 
        Final_ID = []
        for c in classified_Trackers.keys():
            Trackers = classified_Trackers[c]
            if len(Trackers) >= 4:
                for i in range(len(Trackers)-1):
                    Trackers[i].merge(Trackers[-1])
                locs = set(list(Trackers[-1].dets[:,-1]))
                if locs & {1.,2.,3.,4.} == {1.,2.,3.,4.}:
                    Final_ID.append(Trackers[-1])
        del classified_Trackers

        return Final_ID 
if __name__ == '__main__':
    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('tracks_dir', help='tracks obj pickle directory')
    parser.add_argument('output', help='output file')
    parser.add_argument('--dump_dir', default='./tmp', help='folder to dump images')
    parser.add_argument('--cluster_method', default='k-means', type=str, help='cluster methods')
    #parser.add_argument('--log_txt', default='./reid_log.txt', type=str, help='Final tracker log')

    args = parser.parse_args()

    # Create sequence names
    loc = [1] #[1, 2, 3, 4]
    loc_n_seq = [4, 6, 2, 3]
    
    # Load and initialize tracks objs with seq names
    multi_cam_tracks = []
    seq_id = 1
    for i, l in enumerate(loc):
        single_cam_tracks = []
        for n in range(1,loc_n_seq[i]+1):
            name = 'Loc%d_%d' % (l, n)
            pkl_name = os.path.join(args.tracks_dir, '%s.pkl' % name)
            print('Loading %s...' % pkl_name)
            with open(pkl_name, 'rb') as f:
                tracks = pickle.load(f)
            for t in tracks:
                t.assign_seq_id(seq_id, l)
            seq_id += 1
            single_cam_tracks += tracks
        multi_cam_tracks.append(single_cam_tracks)

    # Multi camera matching
    # len of multi_cam_tracks is equal to the number of Locations
    tracks = multi_camera_matching(args, multi_cam_tracks)

    # Output to file
    if tracks == []:
        print('No class meets the constrain')
        sys.exit()
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    dets = np.concatenate([dets[:,:7], -1*np.ones((dets.shape[0],1)), dets[:,[7]]], axis=1)
    np.savetxt(args.output, dets, fmt='%f')

    # Dump imgs
    print('dumping images...')
    for t in tracks:
        dump_imgs(args.dump_dir, t)
