from ReID_CNN.Model_Wrapper import ResNet_Loader 
from track import Track, intersect_test
from clustering import Top_Down, Seed_KMeans 

import argparse
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import pickle
import pathlib
import shutil

def import_pkl(pkl_name):
    """Import pickle file"""
    print('Loading %s...' % pkl_name)
    with open(pkl_name, 'rb') as f:
        tracks = pickle.load(f)
    return tracks

def dump_imgs(output_dir, track):
    """Dump images from img_paths to output_dir/id"""
    output_dir = os.path.join(output_dir, '%06d' % track.id)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for path in track.dump_img_paths():
        p = path.split('/')
        name = p[-2] + '_' + p[-1]
        shutil.copyfile(path, os.path.join(output_dir, name))

def clustering(args, tracks):
    """Cluster tracks"""
    print('clustering %d tracks with %d clusters...' % (len(tracks), args.k))

    features = [t.summarized_feature() for t in tracks]
    features = np.stack(features, axis=0)
    if args.normalize:
        features = normalize(features, axis=1)

    print('start clustering')
    if args.cluster == 'kmeans':
        Cluster = KMeans(n_clusters=args.k,precompute_distances=True,n_jobs=-1,max_iter=500,n_init=15,verbose=1)
    elif args.cluster == 'minibatchkmeans':
        Cluster = MiniBatchKMeans(n_clusters=args.k,max_iter=500,n_init=3,init_size=args.k,verbose=1,max_no_improvement=1,batch_size=1000)
    elif args.cluster == 'top_down':
        Cluster = Top_Down(n_classes=args.k)
    elif args.cluster == 'seed_kmeans':
        Cluster = Seed_KMeans(n_classes=args.k, n_seeds=args.n_seeds)
    else:
        raise ValueError('Unrecognized cluster method!')
    class_output = Cluster.fit_predict(features)
    print('end clustering')
    del features
    return class_output

def multi_camera_matching(opt,MCT):
    print('multi camera matching...')
    if opt.method == 'cluster':
        idxs = []
        tracks = []
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                tracks.append(MCT[i][j])
                idxs.append((i,j))
        
        classes = clustering(opt, tracks)
        with open(os.path.join(opt.output_dir, 'cluster.pkl'), 'wb') as f:
            pickle.dump(classes, f)

        # dict order by clustered class
        classified_Trackers = defaultdict(list)
        for k in range(len(classes)):
            classified_Trackers[classes[k]].append(idxs[k])

        # Check every cluster class 
        Final_ID = []
        for c in classified_Trackers.keys():
            Trackers = classified_Trackers[c]
            if len(Trackers) >= 4:
                locs = set([i for i, j in Trackers])
                if locs & {0,1,2,3} == {0,1,2,3}:
                    Final_ID.append([MCT[i][j] for i, j in Trackers])
        del classified_Trackers
        print('%d qualified clusters!' % len(Final_ID))

        # check constraint: no tracks in same cluster overlap in time
        k = 0
        for idx, t_list in enumerate(Final_ID):
            conflict_idx = check_conflict(t_list)
            if len(conflict_idx) != 0:
                conflict_feat = [t_list[i].summarized_feature() for i in conflict_idx]
                ref_track = [t_list[i] for i in range(len(t_list)) if i not in conflict_idx]
                ref_feat = np.mean([t.summarized_feature() for t in t_list], axis=0)
                dist = [np.linalg.norm(feat-ref_feat) for feat in conflict_feat]

                sort_idx = np.argsort(dist)
                for i in range(len(sort_idx)):
                    current_idx = conflict_idx[sort_idx[i]]
                    temp = ref_track + [t_list[current_idx]]
                    if len(check_conflict(temp)) == 0:
                        ref_track = temp.copy()
                print('%d --> %d' % (len(t_list), len(ref_track)))
                Final_ID[idx] = ref_track
                k += 1
        print('Find conflict in %d clusters!' % k)
        
        for i in range(len(Final_ID)):
            for j in range(len(Final_ID[i])-1):
                Final_ID[i][j].merge(Final_ID[i][-1])
            Final_ID[i] = Final_ID[i][-1]
        return Final_ID
    else:
        raise NotImplementedError('Unrecognized MCT method!')

def check_conflict(tracks):
    """check if there are 2 tracks in the same frame"""
    conflict_idx = []
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i!=j and tracks[i].seq_id() == tracks[j].seq_id() and \
               intersect_test(tracks[i], tracks[j]):
                conflict_idx.append(i)
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    return conflict_idx

def debug(dets, loc_seq_id):
    """Constrain check for final detections"""
    # Check location
    video_id = np.unique(dets[:,0]).tolist()
    seq2loc = {}
    for i, seqs in enumerate(loc_seq_id):
        for seq in seqs:
            seq2loc[seq] = i
    video_id = [seq2loc[id] for id in video_id]
    cond = False
    for i in range(len(loc_seq_id)):
        cond = cond or (i not in video_id)
    if cond:
        print('Does not satisfy location condition!')
        return False

    # Check frame
    frame = {}
    for i in range(dets.shape[0]):
        seq = dets[i,0]
        f = dets[i,1]
        if (seq,f) in frame:
            return False
        frame[(seq,f)] = 1
    return True

if __name__ == '__main__':
    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('tracks_dir', help='tracks obj pickle directory')
    parser.add_argument('output_dir', help='output dir')
    parser.add_argument('--dump_dir', default='./tmp', help='folder to dump images')
    parser.add_argument('--method', help='multi-camera tracking methods: cluster or rank')
    parser.add_argument('--cluster', default='kmeans', type=str, help='cluster methods')
    parser.add_argument('--normalize', action='store_true', help='whether normalize feature or not')
    parser.add_argument('--k', type=int, help='# of clusters')
    args = parser.parse_args()

    # Create sequence names
    loc = [1, 2, 3, 4]
    loc_n_seq = [4, 6, 2, 3]
    
    # Load and initialize tracks objs with seq names
    multi_cam_tracks = []
    seq_id = 1
    loc_seq_id = []
    for i, l in enumerate(loc):
        single_cam_tracks = []
        seqs = []
        for n in range(1,loc_n_seq[i]+1):
            pkl_name = os.path.join(args.tracks_dir, 'Loc%d_%d.pkl' % (l,n))
            tracks = import_pkl(pkl_name)
            for t in tracks:
                t.assign_seq_id(seq_id, l)
            seqs.append(seq_id)
            seq_id += 1
            single_cam_tracks += tracks
        loc_seq_id.append(seqs)
        multi_cam_tracks.append(single_cam_tracks)

    # Multi camera matching
    # len of multi_cam_tracks is equal to the number of Locations
    tracks = multi_camera_matching(args, multi_cam_tracks)

    # Re-index id & final check
    for i,t in enumerate(tracks):
        t.assign_id(i+1)
        assert debug(t.dump(), loc_seq_id), 'something is wrong!!!'

    # Output to file
    if tracks == []:
        print('No class meets the constrain')
        sys.exit()
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    dets = np.concatenate([dets[:,:7], -1*np.ones((dets.shape[0],1)), dets[:,[7]]], axis=1)
    np.savetxt(os.path.join(args.output_dir, 'track3.txt'), dets, fmt='%f')

    # Dump imgs
    print('dumping images...')
    for t in tracks:
        dump_imgs(args.dump_dir, t)
