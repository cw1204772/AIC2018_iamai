from ReID_CNN.Model_Wrapper import ResNet_Loader 
from track import Track, intersect_test
from clustering import Top_Down, Seed_KMeans 

import argparse
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import pickle
import pathlib
import shutil
from tqdm import tqdm
import torch
import copy

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

def compute_distance(combie_list):
    features = []
    for t in combie_list:
        features.append(t.summarized_feature())
    features = torch.nn.functional.normalize(torch.FloatTensor(np.vstack(features))).cuda()
    dis_matrix = -1*torch.mm(features,torch.t(features))
    dis_matrix = dis_matrix.cpu().numpy()
    return dis_matrix

def compute_k_reciprocal(k,arg_sort_matrix,rank_list,delete_idx,now_idx,bar=False):
    rank_list_left = []
    points_of_rank_list = []
    set_rank = set(rank_list)
    if bar == True:
        pbar = tqdm(total=len(rank_list),leave=True)
    for j in rank_list:
        tmp_idx = arg_sort_matrix[j][:2500]
        tmp_idx = [idx for idx in list(tmp_idx) if idx not in delete_idx]
        if now_idx in tmp_idx[:k]: 
            rank_list_left.append(j)
            point = len(set_rank & set(tmp_idx[:k]))
            points_of_rank_list.append(point)
        if bar: pbar.update(1)
    if bar: pbar.close()
    
    rank_list_left = np.array(rank_list_left)[np.argsort(np.array(points_of_rank_list))[::-1]]
    return list(rank_list_left)
        
def re_ranking(dis_matrix,q_idx):
    k = 20
    final_list = []
    print('Sorting distance matrix...')
    arg_sort_matrix = np.argsort(dis_matrix,axis=1)

    for c,i in enumerate(q_idx):
        #print('Compute No.%d query'%(c+1))
        sorted_idx = arg_sort_matrix[i][:2500]
        # filter the query index out
        sorted_idx = [idx for idx in list(sorted_idx) if idx not in q_idx]
        rank_k = sorted_idx[:k]
        ## k-reciprocal
        #print('-->Start K-reciprocal...')
        no_i_q_idx = [idx for idx in q_idx if idx!=i]
        rank_k_left = compute_k_reciprocal(k,arg_sort_matrix,rank_k,no_i_q_idx,i,bar=False)
        #print('Left %d'%(len(rank_k_left)))
        '''
        ##expansion
        print('-->Left %d in rank list. Start expansion...'%(len(rank_k_left)))
        half_k = k//2 
        points = np.zeros(dis_matrix.shape[0])
        pbar = tqdm(total=len(rank_k_left),leave=True)
        for c,j in enumerate(rank_k_left):
            points[j] = k-c
            sorted_idx = arg_sort_matrix[j][:2500]
            # filter the query index out
            sorted_idx = [idx for idx in list(sorted_idx) if idx not in q_idx]
            rank_h_k = sorted_idx[:half_k]
            ## k-reciprocal
            add_j_q_idx = q_idx + [j]
            rank_h_k_left = compute_k_reciprocal(half_k,arg_sort_matrix,rank_h_k,add_j_q_idx,j)
            # give points
            for index in range(len(rank_h_k_left)):
                p = (k-c)-((k-c)/half_k)*(index+1)
                points[rank_h_k_left[index]] += p
            pbar.update(1)
        pbar.close()
        non_zero = np.count_nonzero(points)
        print('Final left %d'%(non_zero))
        final_index = np.argsort(points)[::-1][:non_zero]
        print(final_index)
        '''
        if c%50 == 0: print(c)
    
        final_list.append(rank_k_left)
        #final_list.append(final_index)
    return final_list



def clustering(args, tracks):
    """Cluster tracks"""
    print('clustering %d tracks with %d clusters...' % (len(tracks), args.k))

    features = [t.summarized_feature(args.sum) for t in tracks]
    features = np.stack(features, axis=0)
    if args.normalize:
        features = normalize(features, axis=1)

    print('start clustering')
    if args.cluster == 'kmeans':
        Cluster = KMeans(n_clusters=args.k,precompute_distances=True,n_jobs=-1,max_iter=500,n_init=15,verbose=1)
    elif args.cluster == 'minibatchkmeans':
        Cluster = MiniBatchKMeans(n_clusters=args.k,max_iter=500,n_init=3,init_size=args.k,verbose=1,max_no_improvement=250,batch_size=1000)
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
    """Multi-camera matching"""
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

        # check constraint and merge
        tracks = []
        for t_list in tqdm(Final_ID):
            tracks.append(merge_tracks(opt, t_list))
        
        return tracks

    elif args.method == 'bottom_up_cluster':
        all_tracks = []
        # clustering within same location
        for i, loc_tracks in enumerate(MCT):
            print('Loc%d' % (i+1))
            opt.k = len(loc_tracks) // opt.n
            classes = clustering(opt, loc_tracks)
            with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'wb') as f:
                pickle.dump(classes, f)
            #with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'rb') as f:
            #    classes = pickle.load(f)

            for class_id in tqdm(np.unique(classes).tolist()):
                select = np.where(classes == class_id)[0]
                tracks = [loc_tracks[j] for j in select.tolist()]
                track = merge_tracks(opt, tracks)
                assert debug_frame(track.dump())
                all_tracks.append(track)
        
        # clustering with all locations
        opt.k = len(all_tracks) // opt.n
        classes = clustering(opt, all_tracks)
        with open(os.path.join(opt.output_dir, 'all_loc_cluster.pkl'), 'wb') as f:
            pickle.dump(classes, f)
        #with open(os.path.join(opt.output_dir, 'all_loc_cluster.pkl'), 'rb') as f:
        #    classes = pickle.load(f)

        clusters = []
        for class_id in tqdm(np.unique(classes).tolist()):
            select = np.where(classes == class_id)[0]
            tracks = [all_tracks[j] for j in select.tolist()]
            for t in tracks:
                assert debug_frame(t.dump())
            track = merge_tracks(opt, tracks)
            if debug_loc(track.dump(), loc_seq_id):
                clusters.append(track)
        print('%d qualified clusters!' % len(clusters))
        return clusters

    elif opt.method == 're-rank-4' or opt.method == 're-rank-all':
        print('reranking...')
        # First cluster every Location
        Locs = [[],[],[],[]]
        for i,loc_tracks in enumerate(MCT):
            print('Loc%d'%(i+1))
            opt.k = len(loc_tracks)//opt.n
            classes = clustering(opt,loc_tracks)
            with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'wb') as f:
                pickle.dump(classes, f)
            #with open(os.path.join('/home/cw1204772/Dataset/AIC2018/MCT/fasta_joint_vanilla_biased_knn_avg','Loc%d_cluster.pkl'%(i+1)),'rb') as f:
            #    classes = pickle.load(f)

            for class_id in tqdm(np.unique(classes).tolist()):
                select = np.where(classes == class_id)[0]
                tracks = [loc_tracks[j] for j in select.tolist()]
                track = merge_tracks(opt,tracks)
                assert debug_frame(track.dump())
                Locs[i].append(track)
        with open(os.path.join(opt.output_dir,'Locs.pkl'),'wb') as f:
            pickle.dump(Locs,f)
        #with open(os.path.join(opt.output_dir,'Locs.pkl'),'rb') as f:
        #    Locs  = pickle.load(f)
        ###############
        ''' 
        with open(os.path.join(opt.output_dir,'Loc4_cluster.pkl'), 'rb') as f:
            classes = pickle.load(f)

        temp = []
        for class_id in tqdm(np.unique(classes).tolist()):
            select = np.where(classes == class_id)[0]
            tracks = [MCT[3][j] for j in select.tolist()]
            track = merge_tracks(opt,tracks)
            assert debug_frame(track.dump())
            temp.append(track)
        Locs[3] = temp
        with open(os.path.join(opt.output_dir,'Locs.pkl'),'wb') as f:
            pickle.dump(Locs,f)       
        ''' 
        ###############
        
        q_idx = list(range(len(Locs[3])))
        if opt.method == 're-rank-all':
            tracks = Locs[3] + Locs[0] + Locs[1] + Locs[2]
            dis_matrix = compute_distance(tracks)
            rank_list = re_ranking(dis_matrix, q_idx)
            
            outputs = []
            for i, ranks in enumerate(rank_list):
                track = tracks[q_idx[i]]
                for idx in ranks[:opt.n]:
                    track = priority_merge(track, tracks[idx])
                if debug_loc(track.dump(), loc_seq_id):
                    outputs.append(track)
            return outputs
            
        elif opt.method == 're-rank-4':
            
            q_idx = list(range(len(Locs[3])))
            multi_loc_tracks = []
            for i in [0,1,2]:
                tracks = Locs[3] + Locs[i]
                dis_matrix = compute_distance(tracks)
                rank_list = re_ranking(dis_matrix, q_idx)
                with open(os.path.join(opt.output_dir, 're_rank%d.pkl' % i), 'wb') as f:
                    pickle.dump(rank_list, f)
                #with open(os.path.join(opt.output_dir, 're_rank%d.pkl' % i), 'rb') as f:
                #    rank_list = pickle.load(f)

                #rank_list = [[len(tracks)-1], [], [len(tracks)-1,len(tracks)-2]]
                #for i in range(len(q_idx)-3):
                #    rank_list.append([])
                assert len(rank_list)==len(q_idx)
                
                single_loc_tracks = []
                for j, ranks in enumerate(rank_list):
                    #print(ranks)
                    if len(ranks)==0:
                        single_loc_tracks.append(None)
                        #print('None')
                    else:
                        track = copy.copy(tracks[ranks[0]])
                        if len(ranks) > 1:
                            for idx in ranks[1:]:
                                track = priority_merge(track, tracks[idx])
                                if tracks[idx].dump() is None: 
                                    sys.exit('not right!')
                        #print(track.dump())
                        single_loc_tracks.append(track)
                        if single_loc_tracks[-1].dump() is None: 
                            print(i,j)
                            print(ranks)
                            sys.exit('not right!')
                    #if j == 2:
                    #    print(single_loc_tracks[-1].dump())
                    #    sys.exit()
                    #if j == 5: sys.exit()
                multi_loc_tracks.append(single_loc_tracks)
            with open(os.path.join(opt.output_dir, 'after_re_ranking.pkl'), 'wb') as f:
                pickle.dump(multi_loc_tracks, f)
            
            #with open(os.path.join(opt.output_dir, 'after_re_ranking.pkl'), 'rb') as f:
            #    multi_loc_tracks = pickle.load(f)
            
            
            #print(multi_loc_tracks[0][2].dump())

            outputs = []
            for i, idx in enumerate(q_idx):
                track = Locs[3][idx]
                for j in range(len(multi_loc_tracks)):
                    #print('--------------------')
                    #print(j,i)
                    #if multi_loc_tracks[j][i].dump() is None: 
                    #print(j,i)
                    #print(multi_loc_tracks[j][i] is not None)
                    #print(multi_loc_tracks[j][i].dump().shape)
                    if multi_loc_tracks[j][i] is not None:
                        #print(multi_loc_tracks[j][i].dump())
                        #print(multi_loc_tracks[j][i].dump().shape)
                        #print(track.dump().shape)
                        multi_loc_tracks[j][i].merge(track)
                    #print(multi_loc_tracks[0][2].dump())
                if debug_loc(track.dump(), loc_seq_id):
                    outputs.append(track)
            return outputs

    elif opt.method == 'biased_knn':
        
        ref_tracks = []
        tgt_tracks = []
        # clustering within same location
        for i, loc_tracks in enumerate(MCT):
            print('Loc%d' % (i+1))
            opt.k = len(loc_tracks) // opt.n
            classes = clustering(opt, loc_tracks)
            with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'wb') as f:
                pickle.dump(classes, f)
            #with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'rb') as f:
            #    classes = pickle.load(f)

            for class_id in tqdm(np.unique(classes).tolist()):
                select = np.where(classes == class_id)[0]
                tracks = [loc_tracks[j] for j in select.tolist()]
                track = merge_tracks(opt, tracks)
                assert debug_frame(track.dump())
                if i == 3: ref_tracks.append(track)
                else: tgt_tracks.append(track)
        
        #with open(os.path.join(opt.output_dir, 'ref_tracks.pkl'), 'wb') as f:
        #    pickle.dump(ref_tracks, f)
        #with open(os.path.join(opt.output_dir, 'tgt_tracks.pkl'), 'wb') as f:
        #    pickle.dump(tgt_tracks, f)
        #with open(os.path.join(opt.output_dir, 'ref_tracks.pkl'), 'rb') as f:
        #    ref_tracks = pickle.load(f)
        #with open(os.path.join(opt.output_dir, 'tgt_tracks.pkl'), 'rb') as f:
        #    tgt_tracks = pickle.load(f)

        # K nearest neighbor classifier
        features = []
        targets = []
        for i, t in enumerate(ref_tracks):
            features.append(t.summarized_feature())
            targets.append(i)
        features = np.stack(features, axis=0)
        targets = np.array(targets)
        print('start knn...')
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        knn.fit(features, targets)

        features = []
        for t in tgt_tracks:
            features.append(t.summarized_feature(opt.sum))
        features = np.stack(features, axis=0)
        print('predicting...')
        classes = knn.predict(features)
        
        for class_id in tqdm(np.unique(classes).tolist()):
            select = np.where(classes == class_id)[0]
            tracks = [tgt_tracks[i] for i in select.tolist()]
            track = merge_tracks(opt, tracks, [ref_tracks[class_id]])
            ref_tracks[class_id] = merge_tracks(opt, [ref_tracks[class_id], track])
        
        # Filter out tracks that does not appear in all 4 locations
        outputs = []
        for t in ref_tracks:
            if debug_loc(t.dump(), loc_seq_id):
                outputs.append(t)
        print('%d qualified clusters!' % len(outputs))
        return outputs

    elif opt.method == 'biased_biased_knn':
        # cluster with loc 3 & 4
        ref_tracks = []
        for i in range(2,3+1):
            ref_tracks += MCT[i]
        opt.k = 2000
        ref_classes = clustering(opt, ref_tracks)
        with open('ref_classes.pkl', 'wb') as f:
            pickle.dump(ref_classes, f)
        #with open('ref_classes.pkl', 'rb') as f:
        #    ref_classes = pickle.load(f)
        
        # build knn
        features = [t.summarized_feature() for t in ref_tracks]
        features = np.stack(features, axis=0)
        print('start knn...')
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        knn.fit(features, ref_classes)
        with open('train_knn.pkl', 'wb') as f:
            pickle.dump(knn, f)
        #with open('train_knn.pkl', 'rb') as f:
        #    knn = pickle.load(f)

        # classify loc 1 & 2 with knn
        tgt_tracks = []
        for i in range(0,1+1):
            tgt_tracks += MCT[i]
        features = [t.summarized_feature() for t in tgt_tracks]
        features = np.stack(features, axis=0)
        print('predicting...')
        tgt_classes = knn.predict(features)
        with open('tgt_classes.pkl', 'wb') as f:
            pickle.dump(tgt_classes, f)
        #with open('tgt_classes.pkl', 'rb') as f:
        #    tgt_classes = pickle.load(f)

        # merge and check constraint
        outputs = []
        for class_id in tqdm(np.unique(ref_classes).tolist()):
            ref_select = np.where(ref_classes == class_id)[0]
            ref_t = [ref_tracks[i] for i in ref_select.tolist()]
            tgt_select = np.where(tgt_classes == class_id)[0]
            tgt_t = [tgt_tracks[i] for i in tgt_select.tolist()]
            track = merge_tracks(opt, ref_t+tgt_t, ref_t)
            if debug_loc(track.dump(), loc_seq_id):
                outputs.append(track)
        print('%d qualified clusters!' % len(outputs))
        return outputs
        '''
        elif opt.method == 'hac':
            tracks = []
            # clustering within same location
            for i, loc_tracks in enumerate(MCT):
                print('Loc%d' % (i+1))
                opt.k = len(loc_tracks) // opt.n
                #classes = clustering(opt, loc_tracks)
                #with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'wb') as f:
                #    pickle.dump(classes, f)
                with open(os.path.join(opt.output_dir, 'Loc%d_cluster.pkl' % (i+1)), 'rb') as f:
                    classes = pickle.load(f)
    
                for class_id in tqdm(np.unique(classes).tolist()):
                    select = np.where(classes == class_id)[0]
                    tracks = [loc_tracks[j] for j in select.tolist()]
                    track = merge_tracks(opt, tracks)
                    assert debug_frame(track.dump())
                    tracks.append(track)
            
            # HAC
            features = [t.summarized_feature(opt.sum) for t in tracks]
            #knn_graph = kneighbors_graph(features, 30, include_self=False, n_jobs=-1)
            #with open(os.path.join(opt.output_dir, 'knn_graph.pkl'), 'wb') as f:
            #    pickle.dump(knn_graph, f)
            hac = AgglomerativeClustering(linkage='ward',
                                          connectivity=None,
                                          n_clusters=opt.k)
            classes = hac.fit_predict(features)
            with open(os.path.join(opt.output_dir, 'hac_classes.pkl'), 'wb') as f:
                pickle.dump(classes, f)
    
            # merge and check constraint
            outputs = []
            for class_id in tqdm(np.unique(classes).tolist()):
                select = np.where(classes == class_id)[0]
                t = [tracks[i] for i in select.tolist()]
                track = merge_tracks(opt, t)
                if debug_loc(track.dump(), loc_seq_id):
                    outputs.append(track)
            print('%d qualified clusters!' % len(outputs))
            return outputs
        '''
    else:
        raise NotImplementedError('Unrecognized MCT method!')

def merge_tracks(args, tracks, feat_tracks=None):
    """Check frame conflict & merge tracks"""
    # Check frame conflict and fix it
    if feat_tracks is None: feat_tracks = tracks
    dets = np.concatenate([t.dump() for t in tracks], axis=0)
    if not debug_frame(dets):
        conflict_idx = check_conflict(tracks)
        conflict_feat = [tracks[i].summarized_feature(args.sum) for i in conflict_idx]
        ref_track = [tracks[i] for i in range(len(tracks)) if i not in conflict_idx]
        ref_feat = np.mean([t.summarized_feature(args.sum) for t in feat_tracks], axis=0)
        dist = [np.linalg.norm(feat-ref_feat) for feat in conflict_feat]

        sort_idx = np.argsort(dist)
        for i in range(len(sort_idx)):
            current_idx = conflict_idx[sort_idx[i]]
            temp = ref_track + [tracks[current_idx]]
            dets = np.concatenate([t.dump() for t in temp], axis=0)
            if debug_frame(dets):
                ref_track = temp.copy()
        #print('%d --> %d' % (len(tracks), len(ref_track)))
        tracks = ref_track
    
    # Merge tracks
    for j in range(len(tracks)-1):
        tracks[j].merge(tracks[-1])
    assert debug_frame(tracks[-1].dump())
    return tracks[-1]

def check_conflict(tracks):
    """Check if there are 2 tracks in the same frame"""
    conflict_idx = []
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            temp_dets = np.concatenate([tracks[i].dump(), tracks[j].dump()], axis=0)
            if i!=j and not debug_frame(temp_dets):
            #if i!=j and tracks[i].seq_id() == tracks[j].seq_id() and \
            #   intersect_test(tracks[i], tracks[j]):
                conflict_idx.append(i)
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    return conflict_idx

def remove(args, tracks):
    """Remove detections from tracks"""
    filter_dets = np.loadtxt(args.filter)
    d = {tuple(row[[0,1,3,4]].astype(int)):True for row in filter_dets}
    del_idx = []
    for i,t in enumerate(tracks):
        dets = t.dump()
        temp = []
        for j in range(dets.shape[0]):
            k = tuple(dets[j,[0,1,3,4]].astype(int))
            if k not in d:
                temp.append(dets[j,:])
        temp = np.stack(temp, axis=0)
        if not debug_loc(temp, loc_seq_id):
            del_idx.append(i)
    print('removing %d tracks by some hack!' % len(del_idx))
    for i in reversed(del_idx):
        del tracks[i]
    return tracks

def fill(tracks, n):
    """Squeeze tracks into n tracks"""
    scores = []
    for t in tracks:
        mean_feature = np.mean(t.features, axis=0).reshape(1,-1)
        norms = np.linalg.norm((t.features - mean_feature), axis=1)
        scores.append(np.mean(norms))
    sample = np.argsort(scores)

    outputs = []
    for i,idx in enumerate(sample.tolist()):
        if i < n:
            outputs.append(tracks[idx])
        else:
            outputs[i%n] = priority_merge(outputs[i%n], tracks[idx])
    return outputs

def cluster_fill(args, tracks, n):
    scores = []
    for t in tracks:
        mean_feature = np.mean(t.features, axis=0).reshape(1,-1)
        norms = np.linalg.norm((t.features - mean_feature), axis=1)
        scores.append(np.mean(norms))
    sample = np.argsort(scores)
    tracks = [tracks[i] for i in sample.tolist()]

    args.k = n
    args.cluster = 'kmeans'
    classes = clustering(args, tracks)
    
    outputs = []
    for class_id in range(n):
        select = np.where(classes == class_id)[0]
        temp_track = tracks[select[0]]
        if select.shape[0] > 1:
            for i in select[1:].tolist():
                temp_track = priority_merge(temp_track, tracks[i])
        outputs.append(temp_track)
    return outputs

def priority_merge(major_track_, minor_track_):
    """Merge 2 tracks with priority"""
    major_track = copy.copy(major_track_)
    minor_track = copy.copy(minor_track_)
    d = {tuple(row[0:2]):True for row in major_track.dump()}
    dump_dets = minor_track.dump()
    dets = minor_track.dets
    valid_dets = []
    for i in range(dump_dets.shape[0]):
        k = tuple(dump_dets[i,0:2])
        if k not in d:
            valid_dets.append(dets[i])
    if len(valid_dets) == 0: minor_track.dets = np.zeros((0,9))
    else: minor_track.dets = np.stack(valid_dets, axis=0)
    minor_track.merge(major_track)
    assert debug_frame(major_track.dump())
    return major_track

def sample_tracks(tracks, n):
    """Sample top n most densly-clustered tracks"""
    scores = []
    for t in tracks:
        mean_feature = np.mean(t.features, axis=0).reshape(1,-1)
        norms = np.linalg.norm((t.features - mean_feature), axis=1)
        scores.append(np.mean(norms))
    sample = np.argsort(scores)[:n]
    return [tracks[i] for i in sample.tolist()]

def debug_loc(dets, loc_seq_id):
    """Check location constraint"""
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
        return False
    return True

def debug_frame(dets):
    """Check frame constraint"""
    frame = {}
    for i in range(dets.shape[0]):
        seq = dets[i,0]
        f = dets[i,1]
        if (seq,f) in frame:
            return False
        frame[(seq,f)] = 1
    return True

def debug_id(dets):
    """Check id constraint"""
    id = np.unique(dets[:,2])
    if np.max(id) > 100 or np.min(id) < 1:
        return False
    return True

if __name__ == '__main__':
    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('tracks_dir', help='tracks obj pickle directory')
    parser.add_argument('output_dir', help='output dir')
    parser.add_argument('--dump_dir', default='./tmp', help='folder to dump images')
    parser.add_argument('--method', help='multi-camera tracking methods: cluster or bottom_up_cluster or rank')
    parser.add_argument('--cluster', default='kmeans', type=str, help='cluster methods')
    parser.add_argument('--normalize', action='store_true', help='whether normalize feature or not')
    parser.add_argument('--k', type=int, help='# of clusters')
    parser.add_argument('--n', type=int, help='bottom up parameter')
    parser.add_argument('--sum', default='avg', help='feature summarization method: max or avg')
    parser.add_argument('--filter', help='the filter file for filtering')
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
    with open(os.path.join(args.output_dir, 'after_mct.pkl'), 'wb') as f:
        pickle.dump(tracks, f)
    #with open(os.path.join(args.output_dir, 'after_mct.pkl'), 'rb') as f:
    #    tracks = pickle.load(f)
    #print(len(tracks))
    #sys.exit()

    # Remove detections with known submissions
    if args.filter is not None:
        tracks = remove(args, tracks)
        with open(os.path.join(args.output_dir, 'after_remove.pkl'), 'wb') as f:
            pickle.dump(tracks, f)

    # Decide the final 100 tracks
    #tracks = sample_tracks(tracks, 100)
    #tracks = sample_tracks(tracks, 300)[200:300]
    #tracks = fill(tracks, 100)
    tracks = cluster_fill(args, tracks, 100)

    # Re-index id & final check
    for i,t in enumerate(tracks):
        t.assign_id(i+1)
        if not debug_loc(t.dump(), loc_seq_id):
            sys.exit('Does not satisfy location condition!')
        if not debug_frame(t.dump()): 
            sys.exit('Does not satisfy frame condition!')
        if not debug_id(t.dump()):
            sys.exit('Does not satisfy object id condition')

    # Output to file
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    dets = np.concatenate([dets[:,:7], -1*np.ones((dets.shape[0],1)), dets[:,[7]]], axis=1)
    dets[:,5:7] = dets[:,5:7] + dets[:,3:5]
    np.savetxt(os.path.join(args.output_dir, 'track3.txt'), 
               dets, fmt='%d %d %d %d %d %d %d %d %f')

    # Dump imgs
    print('dumping images...')
    for t in tracks:
        dump_imgs(args.dump_dir, t)
