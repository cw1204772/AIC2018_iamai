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

def parse_tracks(tracking_csv):
    """Read tracking csv to Track obj"""
    print('parsing tracks...')
    dets = np.loadtxt(tracking_csv, delimiter=',')

    # Split detections by id 
    sort_idx = np.argsort(dets[:, 1])
    dets = dets[sort_idx, :7]
    det_blocks = np.split(dets, np.where(np.diff(dets[:, 1]))[0]+1, axis=0)

    tracks = []
    for b in det_blocks:
        tracks.append(Track(b))
    return tracks

def extract_images(tracks, video, sample_interval, save_dir):
    """Extract images for each track"""
    print('extracting images...')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    if len(os.listdir(save_dir)) != 0:
        print('temp_dir is not empty, using already extracted images!')
        return

    # Gather information of images
    imgs_info = []
    for t in tracks:
        imgs_info.append(t.sample_dets(sample_interval))
    imgs_info = np.concatenate(imgs_info, axis=0)
    imgs_info[:, 4:6] += imgs_info[:, 2:4] # x0,y0,x1,y1
    n_imgs = imgs_info.shape[0]

    # Sort image info by frames
    sort_idx = np.argsort(imgs_info[:, 0])
    imgs_info = imgs_info[sort_idx, :]
    imgs_info = np.split(imgs_info, np.where(np.diff(imgs_info[:, 0]))[0]+1, axis=0)

    # Extract imgs
    video = cv2.VideoCapture(video)
    framenumber = 0
    i = 0
    k = 0
    pbar = ProgressBar(max_value=n_imgs)
    while True:
        ret, frame = video.read()
        if not ret: break
        if i == len(imgs_info): break
        if framenumber == imgs_info[i][0,0]:
            frame = frame[:,:,[2,1,0]]
            for j in range(imgs_info[i].shape[0]):
                id = imgs_info[i][j, 1].astype(int)
                x0, y0 = imgs_info[i][j, 2:4].astype(int)
                x1, y1 = imgs_info[i][j, 4:6].astype(int)
                crop = frame[y0:y1, x0:x1]
                assert id<1e+7 and framenumber<1e+7
                img_name = os.path.join(save_dir, '%06d_%06d.jpg' % (id, framenumber))
                imsave(img_name, crop)
                pbar.update(k)
                k += 1
            i += 1
        framenumber += 1
    pbar.finish()

def extract_features(tracks, temp_dir, reid_model, n_layers, batch_size):
    """Extract features, then import features back into tracks"""
    print('extracting features...')
    img_names = sorted(os.listdir(temp_dir))
    img_paths = [os.path.join(temp_dir, img) for img in img_names]
    reid_model = ResNet_Loader(reid_model, n_layers, batch_size)
    features = reid_model.inference(img_paths).numpy()

    # Import features to tracks
    # (Detections should already be sorted by id)
    img_ids = np.array([int(n.split('_')[0]) for n in img_names])
    feature_blocks = np.split(features, np.where(np.diff(img_ids))[0]+1, axis=0)
    assert len(feature_blocks)==len(tracks)
    for i,t in enumerate(tracks):
         t.import_features(feature_blocks[i])
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
    parser.add_argument('tracking_csv', help='tracking result csv file')
    parser.add_argument('video', help='location of corresponding video')
    parser.add_argument('output', help='output dir for track obj pickle & track csv')
    parser.add_argument('--int', default=10, type=int, help='feature sampling interval in each track')
    parser.add_argument('--temp_dir', default='./tmp', type=str, help='temp dir for saving img for reid')
    parser.add_argument('--window', default=15, type=int, help='how many frames will tracker search to revive')
    parser.add_argument('--f_th', default=200, type=float, help='feature distance threshold')
    parser.add_argument('--b_th', default=200, type=float, help='bbox distance threshold')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--reid_model', required=True, type=str, help='reid cnn model')
    parser.add_argument('--n_layers', type=int, required=True, help='# of layers of reid_model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for reid cnn model')
    args = parser.parse_args()

    # Read tracks
    tracks = parse_tracks(args.tracking_csv)
   
    # Extract images
    extract_images(tracks, args.video, args.int, args.temp_dir)
    
    # Extract features
    tracks = extract_features(tracks, args.temp_dir, args.reid_model, args.n_layers, args.batch_size)

    # Single camera tracking
    tracks = single_camera_tracking(tracks, args.window, args.f_th, args.b_th, args.verbose)

    # Save track obj
    os.system('mkdir -p %s' % args.output)
    file_name = args.video.split('/')[-1].split('.')[0]
    with open(os.path.join(args.output, '%s.pkl'%file_name), 'wb') as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    np.savetxt(os.path.join(args.output, '%s.csv'%file_name), dets, delimiter=',', fmt='%f')
    #os.system('rm -r %s'%(args.temp_dir))

