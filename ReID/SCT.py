from ReID_CNN.Model_Wrapper import ResNet_Loader 
from track import Track

import argparse
import numpy as np
import cv2
import os
from scipy.misc import imsave
from progressbar import ProgressBar
import sys
import pathlib
import pickle

def import_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def extract_features(tracks, reid_model, n_layers, batch_size):
    """Extract features, then import features back into tracks"""
    print('extracting features...')
    img_paths = []
    img_n = []
    for t in tracks:
        img_path = t.dump_img_paths()
        img_paths += img_path
        img_n.append(len(img_path))
    reid_model = ResNet_Loader(reid_model, n_layers, batch_size)
    features = reid_model.inference(img_paths).numpy()

    # Import features to tracks
    i = 0
    for t, n in zip(tracks, img_n):
        assert n == len(t.dump_img_paths())
        t.import_features(features[i:i+n])
        i += n 
    return tracks

def single_camera_tracking(tracks, window, feature_th, bbox_th, verbose):
    """Single camera tracking"""
    print('single camera tracking...')
    dead_time = [t.dead_time() for t in tracks]
    sorted_idx = np.argsort(dead_time)
    birth_time = np.array([t.birth_time() for t in tracks])

    delete_list = []
    pbar = ProgressBar(max_value=len(tracks))
    for n,i in enumerate(sorted_idx):
        if verbose: print('---------- Track %d -----------' % tracks[i].id)
        t0 = tracks[i].dead_time()
        min_j = -1
        min_score = float('inf')
        candidates = np.where((birth_time>t0) & ((birth_time-t0)<window))[0].tolist()
        for j in candidates:
            score = tracks[i].sct_match(tracks[j], bbox_th)
            if verbose: print('* matches track %d: %f' % (tracks[j].id, score))
            if score < min_score:
                min_j = j
                min_score = score
        if min_j != -1 and min_score < feature_th:
            if verbose: print('===> merge to track %d!' % tracks[min_j].id)
            tracks[i].merge(tracks[min_j])
            birth_time[min_j] = tracks[min_j].birth_time()
            delete_list.append(i)
        pbar.update(n)
    pbar.finish()

    for i in reversed(sorted(delete_list)):
        del tracks[i]
    return tracks

if __name__ == '__main__':

    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', help='post_tracking pkl file')
    parser.add_argument('output', help='output dir for track obj pickle & track csv')
    parser.add_argument('--window', default=15, type=int, help='how many frames will tracker search to revive')
    parser.add_argument('--f_th', default=200, type=float, help='feature distance threshold')
    parser.add_argument('--b_th', default=200, type=float, help='bbox distance threshold')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--reid_model', required=True, type=str, help='reid cnn model')
    parser.add_argument('--n_layers', type=int, required=True, help='# of layers of reid_model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for reid cnn model')
    args = parser.parse_args()
    
    # Import tracks 
    tracks = import_pkl(args.pkl)
    
    # Extract features
    tracks = extract_features(tracks, args.reid_model, args.n_layers, args.batch_size)

    # Single camera tracking
    tracks = single_camera_tracking(tracks, args.window, args.f_th, args.b_th, args.verbose)

    # Save track obj
    os.system('mkdir -p %s' % args.output)
    file_name = args.pkl.split('/')[-1].split('.')[0]
    with open(os.path.join(args.output, '%s.pkl'%file_name), 'wb') as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    np.savetxt(os.path.join(args.output, '%s.csv'%file_name), dets, delimiter=',', fmt='%f')
    
