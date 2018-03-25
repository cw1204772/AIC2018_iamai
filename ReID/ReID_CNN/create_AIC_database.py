import os
import argparse
import numpy as np
import pathlib
from track import Track
import pickle
from scipy.misc import imread

def sample_tracks(dets_, tracks, sample_interval, track_idx_offset):
    """Sample tracks at specific interval"""
    # build a track_id to track_idx dict
    id2idx = {t.id:i+track_idx_offset for i,t in enumerate(tracks)}

    # Split detections by frame
    sort_idx = np.argsort(dets_[:, 0])
    dets = dets_[sort_idx, :]
    det_blocks = np.split(dets, np.where(np.diff(dets[:, 0]))[0]+1, axis=0)
    
    # Sample tracks every sample_interval
    l = []
    for b in det_blocks:
        if b[0, 0] % sample_interval == 0:
            unique_ids = np.unique(b[:, 1]).tolist()
            l.append([id2idx[id] for id in unique_ids if id in id2idx])
    return l

def dump_imgs(tracks):
    """Dump image paths of a track into list"""
    l = []
    for t in tracks:
        img_paths = t.dump_img_paths()
        l.append(img_paths)
    return l

def import_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser(description='Database generator for AIC dataset')
    parser.add_argument('input_dir', help='dir containing csv and pkl file for post processing')
    parser.add_argument('--output_pkl', required=True, help='output pkl listing all database imgs and its label')
    parser.add_argument('--track_sample_interval', default=150, type=int, help='interval of sampling tracks')
    args = parser.parse_args()

    loc = [1,2,3,4]
    n_seqs = [4,6,2,3]
    seq_names = []
    for i,l in enumerate(loc):
         for j in range(1,n_seqs[i]+1):
             seq_names.append('Loc%d_%d' % (l, j))
     
    # Main
    sample_list = []
    track_img_list = []
    for s in seq_names:
        csv_name = os.path.join(args.input_dir, '%s.csv' % s)
        pkl_name = os.path.join(args.input_dir, '%s.pkl' % s)

        print('reading csv...')
        dets = np.loadtxt(csv_name, delimiter=',')
        tracks = import_pkl(pkl_name)
        l = sample_tracks(dets, tracks, args.track_sample_interval, len(track_img_list))
         
        sample_list += l
        track_img_list += dump_imgs(tracks)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump({'samples':sample_list, 'track_imgs':track_img_list}, f, protocol=pickle.HIGHEST_PROTOCOL)


