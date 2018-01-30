from ReID_CNN.Model_Wrapper import ResNet_Loader 

import pandas as pd
import argparse
import numpy as np
import cv2
import os
from scipy.misc import imsave
from progressbar import ProgressBar
import sys
import csv
import torch

# IO
parser = argparse.ArgumentParser()
parser.add_argument('SCT_csv_dir', help='csv file directory')
parser.add_argument('video_dir', help='location of corresponding video')
parser.add_argument('model_path', help='location of models')
parser.add_argument('--img_dir', default='./MCT_img', type=str, help='temp dir for saving img for reid')
parser.add_argument('--cluster_method', default='k-means', type=str, help='cluster methods')
parser.add_argument('--verbose', action='store_true', help='verbose')
opt = parser.parse_args()



mp4_file_list = [name for name in os.listdir(opt.video_dir) if (name.split('.')[1]=='mp4')]
csv_file_list = [name for name in os.listdir(opt.SCT_csv_dir) if (name.split('.')[1]=='csv')]
mp4_file_list.sort()
csv_file_list.sort()
assert(len(mp4_file_list)==len(csv_file_list))


print('Stage 1/3: Generated Samples BBOX image for every video')
os.system('mkdir -p %s' % opt.img_dir)
if os.listdir(opt.img_dir) != []:
    print('--> have processed... ')
else:
    os.system('rm -rf %s/*' % opt.img_dir)
    for csv_file,mp4_file in zip(csv_file_list,mp4_file_list):
        print('--> process %s'%mp4_file)
        ## Read csv
        file = open(os.path.join(opt.SCT_csv_dir,csv_file),'r')
        content = csv.reader(file)
        detections = []
        for row in content:
            detections.append(np.array(row[:6]))
        file.close()
        detections = np.array(detections).astype('int16')
        
        ## split by ID
        sample_detections = []
        splits = np.split(detections, np.where(np.diff(detections[:,1]))[0]+1)
        for i in range(len(splits)):
            sample_detections.append(splits[i][::5])
        detections = np.concatenate(sample_detections,axis=0)
        del splits,sample_detections
        
        sort_idx = np.argsort(detections[:,0])
        detections = detections[sort_idx,:]

        v_img_dir = os.path.join(opt.img_dir,mp4_file.split('.')[0])
        os.system('mkdir -p %s' % v_img_dir )
        ##per frame save bbox img
        framenumber = 0
        video = cv2.VideoCapture(os.path.join(opt.video_dir,mp4_file))
        while True:
            ret,frame = video.read()
            if not ret: break
            frame = frame[:,:,np.array([2,1,0])]
            framenumber +=1
            # check whether this frame has detection
            match = detections[:,0]==(framenumber)
            if np.mean(match) == 0:
                # print('skip frame %d'%(framenumber))
                continue
            bbox = detections[match]
            for i in range(bbox.shape[0]):
                id_ = bbox[i,1]
                x0,y0 = bbox[i,2:4]
                x1,y1 = bbox[i,2:4]+bbox[i,4:6]
                cropped = frame[y0:y1,x0:x1]
                file_name = os.path.join(v_img_dir,'%05dF%05d.jpg'%(id_,framenumber))
                imsave(file_name,cropped)
print('Stage 2/3: Extract re-id feature')
reid_model = ResNet_Loader(opt.model_path,n_layer=18,batch_size=32,output_color=True)
cross_video_features = []
cross_video_IDs = []
cross_video_loc = []
cross_video_colors = []
for v_name in mp4_file_list:
    
    print('--> process %s'%v_name.split('.')[0])
    cross_video_loc.append(v_name.split('.')[0])
    img_dir = os.path.join(opt.img_dir,v_name.split('.')[0])
    imgs_name = os.listdir(img_dir)
    imgs_name.sort()
    
    ID_list = np.array([int(name.split('F')[0]) for name in imgs_name])
    imgs_path = [os.path.join(img_dir,name) for name in imgs_name]
    del imgs_name
    cross_video_IDs.append(sorted(list(set(list(ID_list)))))

    same_ID_idx = np.split(np.array(range(len(ID_list))),np.where(np.diff(ID_list))[0]+1)
    features,colors = reid_model.inference(imgs_path)
    features = features.numpy()
    colors = colors.numpy()

    final_features = []
    final_colors = []
    # avg pool
    for i in range(len(same_ID_idx)):
        id_features = features[same_ID_idx[i]]
        id_colors = colors[same_ID_idx[i]]
        final_features.append(np.mean(id_features,axis=0))
        ## 眾數
        counts = np.bincount(id_colors)
        final_colors.append(np.argmax(counts))
    final_features = np.array(final_features)
    final_colors = np.array(final_colors)
    cross_video_features.append(final_features)
    cross_video_colors.append(final_colors)
print(cross_video_features[0].shape)
print(cross_video_colors[0][:10])
# print(len(cross_video_features),cross_video_features[0].shape,cross_video_features[1].shape)
# print(len(cross_video_IDs[0]),len(cross_video_IDs[1]))
# print(cross_video_loc)
# exit(-1)
print('Stage 3/3: Clustering the feature')

if opt.cluster_method == 'k-means':
    from sklearn.cluster import KMeans
    n_tracklet = sum([features.shape[0] for features in cross_video_features])
    Cluster = KMeans(n_clusters=n_tracklet//2,precompute_distances=True,n_jobs=-1)
    output = Cluster.fit_predict(np.concatenate(cross_video_features,axis=0))
    print(output)







