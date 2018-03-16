import os
import argparse
import numpy as np
import pathlib
from track import Track
import pickle

def parse_tracks(dets_):
    """Read tracking csv to Track obj"""
    print('parsing tracks...')

    # Split detections by id 
    sort_idx = np.argsort(dets_[:, 1])
    dets = dets_[sort_idx, :7]
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
                assert id<1e+7 and framenumber<1e+7
                img_name = os.path.join(save_dir, '%06d_%06d.jpg' % (id, framenumber))
                imsave(img_name, crop)
                pbar.update(k)
                k += 1
            i += 1
        framenumber += 1
    pbar.finish()

def import_images(tracks, save_dir):
    """Import image names into tracks objs"""
    print('importing images...')
    img_names = sorted(os.listdir(save_dir))
    img_ids = np.array([int(n.split('_')[0]) for n in img_names])
    img_names = [str(pathlib.Path(os.path.join(save_dir, n)).resolve()) for n in img_names]
    img_names = np.split(img_names, np.where(np.diff(img_ids))[0]+1, axis=0)
    assert len(tracks) == len(img_names)
    for i,t in enumerate(tracks):
        assert len(img_names[i].tolist()) >= 2
        t.import_img_paths(img_names[i].tolist())
    return tracks

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
            l.append([id2idx[id] for id in unique_ids])
    return l

def dump_imgs(tracks):
    """Dump image paths of a track into list"""
    l = []
    for t in tracks:
        l.append(t.dump_img_paths())
    return l

if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser(description='Database generator for AIC dataset')
    parser.add_argument('--csv_dir', required=True, help='dir containing tracking csv for all sequence')
    parser.add_argument('--video_dir', required=True, help='dir containing video for all sequence')
    parser.add_argument('--img_dir', required=True, help='output dir for dataset imgs')
    parser.add_argument('--output_pkl', required=True, help='output pkl listing all database imgs and its label')
    parser.add_argument('--image_sample_interval', default=10, type=int, help='interval of sampling images')
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
        csv_name = os.path.join(args.csv_dir, '%s.csv' % s)
        video_name = os.path.join(args.video_dir, '%s.mp4' % s)
        img_dir = os.path.join(args.img_dir, '%s_tmp' % s)

        print('reading csv...')
        dets = np.loadtxt(csv_name, delimiter=',')
        tracks = parse_tracks(dets)
        extract_images(tracks, video_name, args.image_sample_interval, img_dir)
        tracks = import_images(tracks, img_dir)
        l = sample_tracks(dets, tracks, args.track_sample_interval, len(track_img_list))
         
        sample_list += l
        track_img_list += dump_imgs(tracks)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump({'samples':sample_list, 'track_imgs':track_img_list}, f, protocol=pickle.HIGHEST_PROTOCOL)

    '''         
    with open(args.output_pkl, 'rb') as f:
        d = pickle.load(f)
    sample_list = d['samples']
    track_img_list = d['track_imgs']
    print(len(sample_list))
    exit(-1)
    from PIL import Image
    i = 0
    for s in sample_list:
        print(s)
        for t in s:
            print(track_img_list[t])
            for j in track_img_list[t]:
                img = Image.open(j)
                img.save('tmp/%d.jpg' % i)
                i += 1
        break
    ''' 
