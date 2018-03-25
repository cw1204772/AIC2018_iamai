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

def filter_detections(dets, size_th, mask_file):
    """filter detections"""
    # 1. Check size
    select = (dets[:,4] > size_th) | (dets[:,5] > size_th)
    # 2. Check border
    select = select & \
             (dets[:,2] > 0+20) & ((dets[:,2] + dets[:,4]) < 1920-20) & \
             (dets[:,3] > 0+20) & ((dets[:,3] + dets[:,5]) < 1080-20)
    if mask_file is None:
        return select
    # 3. Check mask
    f = open(mask_file, 'r')
    for line in f:
        x1, y1, x2, y2 = [float(v) for v in line.split()]
        select = select & \
                 ((dets[:,2]+dets[:,4] < x1) | (dets[:,2] > x2) | \
                 (dets[:,3]+dets[:,5] < y1) | (dets[:,3] > y2))
    return select

def sample_detections(dets_, size_th, dist_th, mask_file):
    """Sample detections"""
    # 1. filter detections that exceed border or too small
    select = filter_detections(dets_, size_th, mask_file)
    dets = dets_[select, :]
    assert dets.shape[0] > 0
    # 2. select detections progressively according to distance threshold
    output = []
    last_center = dets[0,2:4] + dets[0,4:6]/2
    output.append(dets[0,:])
    for i in range(1,dets.shape[0]):
        current_center = dets[i,2:4] + dets[i,4:6]/2
        if np.linalg.norm(current_center-last_center) > dist_th:
            output.append(dets[i,:])
            last_center = current_center
    # 3. also select the detection with larget area
    area = dets[:,4] * dets[:,5]
    i = np.argmax(area)
    output.append(dets[i,:])
    return np.stack(output, axis=0)
    
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

def filter_tracks(tracks, size_th, mask_file):
    """Filter tracks that doesn't fulfill requirement""" 
    print('filtering tracks...')
    del_idx = []
    for i,t in enumerate(tracks):
        dets = t.dump()
        if not np.any(filter_detections(dets, size_th, mask_file)):
            del_idx.append(i)

    print('deleting %d tracks...' % len(del_idx))
    for i in reversed(del_idx):
         del tracks[i]
    return tracks

def extract_images(tracks, video, size_th, dist_th, mask_file, save_dir):
    """Extract images for each track"""
    print('extracting images...')
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = str(save_dir.resolve())
    if len(os.listdir(save_dir)) != 0:
        print('temp_dir is not empty, using already extracted images!')
        return

    # Gather information of images
    imgs_info = []
    imgs_path = []
    for t in tracks:
        img_info = sample_detections(t.dump(), size_th, dist_th, mask_file)
        assert np.any(img_info[:,0:2]<1e+7)
        img_path = [os.path.join(save_dir, '%06d_%06d.jpg' % (e[1], e[0])) \
                    for e in img_info.tolist()]
        t.import_img_paths(img_path)
        imgs_info.append(img_info)
        imgs_path += img_path
    imgs_info = np.concatenate(imgs_info, axis=0)
    imgs_info[:, 4:6] += imgs_info[:, 2:4] # x0,y0,x1,y1
    imgs_path = np.array(imgs_path, dtype=str)
    n_imgs = imgs_info.shape[0]

    # Sort image info by frames
    sort_idx = np.argsort(imgs_info[:, 0])
    imgs_info = imgs_info[sort_idx, :]
    imgs_info = np.split(imgs_info, np.where(np.diff(imgs_info[:, 0]))[0]+1, axis=0)
    imgs_path = imgs_path[sort_idx]

    # Extract imgs
    video = cv2.VideoCapture(video)
    framenumber = 1
    i = 0
    k = 0
    pbar = ProgressBar(max_value=n_imgs)
    while True:
        if i == len(imgs_info): break
        ret, frame = video.read()
        if not ret: raise RuntimeError('There are still images not extracted, but video ended!')
        if framenumber == imgs_info[i][0,0]:
            frame = frame[:,:,[2,1,0]]
            for j in range(imgs_info[i].shape[0]):
                id = imgs_info[i][j, 1].astype(int)
                x0, y0 = imgs_info[i][j, 2:4].astype(int)
                x1, y1 = imgs_info[i][j, 4:6].astype(int)
                crop = frame[y0:y1, x0:x1]
                img_name = imgs_path[k]
                imsave(img_name, crop)
                pbar.update(k)
                k += 1
            i += 1
        framenumber += 1
    pbar.finish()
    return tracks

def extract_features(tracks, temp_dir, reid_model, n_layers, batch_size):
    """Extract features, then import features back into tracks"""
    print('extracting features...')
    img_names = sorted(os.listdir(temp_dir))
    img_paths = [str(pathlib.Path(os.path.join(temp_dir, img)).resolve()) for img in img_names]
    reid_model = ResNet_Loader(reid_model, n_layers, batch_size)
    features = reid_model.inference(img_paths).numpy()

    # Import features to tracks
    # (Detections should already be sorted by id)
    img_ids = np.array([int(n.split('_')[0]) for n in img_names])
    feature_blocks = np.split(features, np.where(np.diff(img_ids))[0]+1, axis=0)
    img_paths = np.split(img_paths, np.where(np.diff(img_ids))[0]+1, axis=0)
    assert len(feature_blocks)==len(tracks)
    assert len(img_paths)==len(tracks)
    for i,t in enumerate(tracks):
         t.import_features(feature_blocks[i])
         t.import_img_paths(img_paths[i].tolist())
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
    parser.add_argument('--dist_th', default=140, type=int, help='distance between sampled bbox centers in each track')
    parser.add_argument('--size_th', default=70, type=int, help='filter tracks that do not have detection larger than "filter size" in it longest edge')
    parser.add_argument('--mask', type=str, help='txt that describes where should be masked')
    parser.add_argument('--img_dir', default='./tmp', type=str, help='dir for saving img for reid')
    args = parser.parse_args()

    # Read tracks
    tracks = parse_tracks(args.tracking_csv)

    # Filter tracks
    tracks = filter_tracks(tracks, args.size_th, args.mask)
   
    # Extract images
    tracks = extract_images(tracks, args.video, args.size_th, args.dist_th, args.mask, args.img_dir)
    if tracks is None: 
        sys.exit()
    
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

