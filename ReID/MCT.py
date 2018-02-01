from ReID_CNN.Model_Wrapper import ResNet_Loader 

import pandas as pd
import argparse
import numpy as np
import cv2
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
class Tracker(object):
    def __init__(self,id_,loc,frames,features):
        self.loc = loc
        self.id_ = id_
        self.n_frames = len(frames)
        self.feature = np.mean(features,axis=0)
        self.head_feature = np.mean(features[:math.ceil(self.n_frames/2)],axis=0)
        self.tail_feature = np.mean(features[math.floor(self.n_frames/2):],axis=0)
        self.history = [{'attribute':'self','loc':self.loc,'id':self.id_,'frames':frames}]
    def get_head_frame(self):
        pass
    def get_tail_frame(self):
        pass
    def get_id(self):
        return self.id_
    def get_head_features(self):
        return self.head_feature
    def get_tail_features(self):
        return self.tail_feature
    def get_features(self):
        return self.feature
    def get_n_frames(self):
        return self.n_frames
    def get_info(self):
        return self.history
    def merge_mct_tracker(self,o_tracker):
        info = o_tracker.get_info()
        for i in range(len(info)):
            if info[i]['attribute'] == 'self':
                info[i]['attribute'] = 'mct_start'
        self.history+=info

    def merge_sct_tracker(self,o_tracker):
        info = o_tracker.get_info()[0]
        info['attribute'] = 'sct'
        self.history.append(info)
        
        my_frame = self.n_frames
        other_frame = o_tracker.get_n_frames()
        self.feature = (my_frame/(my_frame+other_frame))*self.feature+\
                       (other_frame/(my_frame+other_frame))*o_tracker.get_features()
        #deal with tail,head feature
        # xxxx
        self.n_frames = my_frame+other_frame

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








def Generate_Track_Image(opt,mp4_file_list,csv_file_list):
    print('Stage 1/3: Generated Samples BBOX image for every video')
    os.system('mkdir -p %s' % opt.img_dir)
    if os.listdir(opt.img_dir) != []:
        print('--> have processed... ')
    else:
        os.system('rm -rf %s/*' % opt.img_dir)
        for csv_file,mp4_file in zip(csv_file_list,mp4_file_list):
            print('--> process %s'%mp4_file)
            ## Read csv
            erase_index = []
            detections = np.loadtxt(os.path.join(opt.track_csv_dir,csv_file), delimiter=',')[:,:6]
            detections = np.array(detections).astype('int16')
            for i in range(detections.shape[0]):
                if detections[i][4] <50 or detections[i][5] <50:
                    erase_index.append(i)
            detections = np.delete(detections,erase_index,axis=0)
            ## split by ID
            sample_detections = []
            splits = np.split(detections, np.where(np.diff(detections[:,1]))[0]+1)
            for i in range(len(splits)):
                sample_detections.append(splits[i][::5])
            detections = np.concatenate(sample_detections,axis=0)
            del splits,sample_detections
            ## sort detections by frame 
            sort_idx = np.argsort(detections[:,0])
            detections = detections[sort_idx,:]

            v_img_dir = os.path.join(opt.img_dir,mp4_file.split('.')[0])
            os.system('mkdir -p %s' % v_img_dir )
            
            ##per frame save bbox img
            video = cv2.VideoCapture(os.path.join(opt.video_dir,mp4_file))
            if cv2.__version__[0] == '3':
               frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            else:
               frame_count = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            framenumber = 0

            ## split by frame: get a list
            detections = np.split(detections,np.where(np.diff(detections[:,0]))[0]+1)
            ptr = 0

            pbar = tqdm(total=int(frame_count),ncols=120,leave=Truegg)
            while True:
                ret,frame = video.read()
                if not ret: break
                pbar.update(1)
                frame = frame[:,:,np.array([2,1,0])]
                framenumber +=1
                # check whether this frame has detection
                if ptr >= len(detections):
                    continue
                if detections[ptr][0,0]!=framenumber:
                    continue
                bbox = detections[ptr]
                ptr+=1
                for i in range(bbox.shape[0]):
                    id_ = bbox[i,1]
                    x0,y0 = bbox[i,2:4]
                    x1,y1 = bbox[i,2:4]+bbox[i,4:6]
                    cropped = frame[y0:y1,x0:x1]
                    file_name = os.path.join(v_img_dir,'%05dF%05d.jpg'%(id_,framenumber))
                    imsave(file_name,cropped)
            pbar.close()

def Extract_Feature_and_SCT(opt,mp4_file_list):
    print('Stage 2/3: Extract re-id feature')
    ReID_model = ResNet_Loader(opt.model_path,n_layer=18,batch_size=32,output_color=False)
    Multi_Cam_Trackers = []
    for v_name in mp4_file_list:
        Loc = v_name.split('.')[0]
        print('--> process %s'%Loc)
        img_dir = os.path.join(opt.img_dir,v_name.split('.')[0])
        imgs_name = os.listdir(img_dir)
        imgs_name.sort() #sort by ids
        
        #Read csv file
        #save bbox to tracker,start frame,end frame etc.. 
        
        #inference
        print('--> Inferencing imgs')
        imgs_path = [os.path.join(img_dir,name) for name in imgs_name]
        features = ReID_model.inference(imgs_path).numpy()
        
        SCT_trackers = {}
        ID_list = np.array([int(name.split('F')[0]) for name in imgs_name])
        frame_list = np.array([int(name.split('F')[1].split('.')[0]) for name in imgs_name])
        same_ID_idx = np.split(np.array(range(len(ID_list))),np.where(np.diff(np.array(ID_list)))[0]+1)
        for i in range(len(same_ID_idx)):
            id_ = ID_list[same_ID_idx[i]][0]
            id_frames = frame_list[same_ID_idx[i]]
            id_features = features[same_ID_idx[i]]
            tracker = Tracker(id_,Loc,id_frames,id_features)
            SCT_trackers[id_] = tracker

        # sort_id_by_dead_time = list(SCT_trackers.keys())
        # sort_id_by_dead_time.sort(key=lambda x:SCT_trackers[x].get_tail_frame())
        
        #SCT
        '''
        while main_idx < len(sort_id_by_dead_time):
            main_tracker = SCT_trackers[sort_id_by_dead_time[main_idx]]
            scores = []
            idxs = []
            for idx  in sort_id_by_dead_time:
                comp_tracker = SCT_trackers[idx]
                if main_tracker.get_id() != comp_tracker.get_id() and \
                        abs(main_tracker.get_tail_frame()-comp_tracker.get_head_frame()) < opt.window:
                    # match 2 tracker
                    scores.append(main_tracker.match_tracker(comp_tracker))
        '''
        # Remember to delete Tracker in "SCT_trackers"
        # After SCT
        # no need dictionary
        SCT_trackers = list(SCT_trackers.values())
        Multi_Cam_Trackers.append(SCT_trackers)
    return Multi_Cam_Trackers


def MCT_matching(opt,MCT):
    print('Stage 3/3: Clustering the feature')
    if opt.cluster_method == 'k-means':
        from sklearn.cluster import KMeans
        # decide n_class to be clustered 
        n_tracker = sum([len(loc_tracker) for loc_tracker in MCT])
        n_classes = n_tracker - 100 * (len(MCT)-1)
        
        all_features = []
        for i in range(len(MCT)):
            for j in range(len(MCT[i])):
                all_features.append(MCT[i][j].get_features())
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
                for t in range(1,len(Trackers)):
                    Trackers[0].merge_mct_tracker(Trackers[t])
                classified_Trackers[c] = Trackers[0]
            else:
                classified_Trackers[c] = Trackers[0]
        #visualize
        for c,tracker in classified_Trackers.items():
            tracker.dump_img(opt,c)
        # dump log 
        file = open(opt.log_txt,'w')
        for c,tracker in classified_Trackers.items():
            tracker.dump_log(file,c)
        file.close()

if __name__ == '__main__':
    # IO
    parser = argparse.ArgumentParser()
    parser.add_argument('track_csv_dir', help='csv file directory')
    parser.add_argument('video_dir', help='location of corresponding video')
    parser.add_argument('model_path', help='location of models')
    parser.add_argument('--img_dir', default='./MCT_img', type=str, help='temp dir for saving img for reid')
    parser.add_argument('--cluster_method', default='k-means', type=str, help='cluster methods')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--window', type=int,default=15, help='tracker check window')
    parser.add_argument('--log_txt', default='./reid_log.txt', type=str, help='Final tracker log')

    opt = parser.parse_args()

    mp4_file_list = [name for name in os.listdir(opt.video_dir) if (name.split('.')[1]=='mp4')]
    csv_file_list = [name for name in os.listdir(opt.track_csv_dir) if (name.split('.')[1]=='csv')]
    mp4_file_list.sort()
    csv_file_list.sort()
    assert(len(mp4_file_list)==len(csv_file_list))
    
    Generate_Track_Image(opt,mp4_file_list,csv_file_list)
    Multi_Cam_Trackers = Extract_Feature_and_SCT(opt,mp4_file_list)
    MCT_matching(opt,Multi_Cam_Trackers)



                

    







