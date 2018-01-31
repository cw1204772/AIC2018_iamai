from ReID_CNN.Model_Wrapper import ResNet_Loader 

import argparse
import numpy as np
import cv2
import os
from scipy.misc import imsave
from progressbar import ProgressBar
import sys

class Tracker(object):
  def __init__(self, id_, save_dir, window_size):
    self.id_ = int(id_)
    self.imgs = []
    self.pos = {}
    self.frames = []
    self.save_dir = save_dir
    self.window_size = window_size
    self.features = {}

  def update(self, img, pos, frame):
    self.imgs.append(img)
    self.pos[frame] = pos
    self.frames.append(frame)
    self.life = 1

  def forward(self):
    if self.life == 0:
      head_end = self.window_size if len(self.frames) >= self.window_size else len(self.frames)
      tail_start = len(self.frames)-self.window_size if len(self.frames) >= self.window_size else 0
      extract_idxs = set(list(range(0, head_end))+list(range(tail_start, len(self.frames))))
      for idx in extract_idxs:
        imsave(os.path.join(self.save_dir, '%08d_F%06d.jpg' % (self.id_, self.frames[idx])),
               self.imgs[idx])
      return True
    self.life -= 1
    return False

  def get_head(self):
    return self.frames[0]
  
  def get_tail(self):
    return self.frames[-1]
  
  def get_id(self):
    return self.id_

  def add_feature(self, feature, frame):
    self.features[frame] = feature

  def get_pos(self, frame):
    return self.pos[frame]

  def get_feature(self, frame):
    return self.features[frame]
  
  def print(self):
    print('tracker id:', self.id_)
    print('frames:', self.frames)
    print('pos:', self.pos)
    print('features:', self.features)

  def write_file(self, f):
    for frame in self.frames:
      f.write('%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n' % 
             (frame, self.id_, 
              self.pos[frame][0], self.pos[frame][1], 
              self.pos[frame][2] - self.pos[frame][0],
              self.pos[frame][3] - self.pos[frame][1]))

def bbox_iou(bb1, bb2):
  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
  bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou

def bbox_dist(pos1, pos2):
  x1 = (pos1[0]+pos1[2])/2.
  x2 = (pos2[0]+pos2[2])/2.
  y1 = (pos1[1]+pos1[3])/2.
  y2 = (pos2[1]+pos2[3])/2.
  return np.sqrt((x1-x2)**2+(y1-y2)**2)

def l2dist(feat1, feat2):
  return np.sum((feat1-feat2)**2)

def match_trackers(tracker1, tracker2):
  if opt.verbose: print('* candidate tracker:', tracker2.get_id())
  if tracker1.get_tail() >= tracker2.get_head() and \
     tracker1.get_head() <= tracker2.get_tail():
    start_frame = tracker2.get_head()
    end_frame = tracker1.get_tail()
    ious = []
    dists = []
    for i in range(start_frame, end_frame+1):
      #print(i)
      if i in tracker1.pos and i in tracker2.pos:
        ious.append(bbox_iou(tracker1.get_pos(i), tracker2.get_pos(i)))
        dists.append(l2dist(tracker1.get_feature(i), tracker2.get_feature(i)))
    if len(ious)==0:
      tracker1.print()
      tracker2.print()
      sys.exit('ERROR')
    if opt.verbose: print('** max iou =', max(ious))
    if max(ious) > 0.1: return min(dists)
    else: return float('inf')
  else:
    if tracker1.get_tail() < tracker2.get_head():
      frame1 = tracker1.get_tail()
      frame2 = tracker2.get_head()
    elif tracker1.get_head() > tracker2.get_tail():
      frame1 = tracker1.get_head()
      frame2 = tracker2.get_tail()
    else:
      sys.exit('ERROR!')
    if opt.verbose: print('** dist =', bbox_dist(tracker1.get_pos(frame1), 
                                                 tracker2.get_pos(frame2)))
    if bbox_dist(tracker1.get_pos(frame1), 
                 tracker2.get_pos(frame2)) < 200:
      return l2dist(tracker1.get_feature(frame1),
                    tracker2.get_feature(frame2))
    else: return float('inf')

def merge_trackers(tracker1, tracker2):
  if tracker1.get_tail() >= tracker2.get_head():
    del_idx = []
    for i, f in enumerate(tracker1.frames):
      if f >= tracker2.get_head():
        del_idx.append(i)
    for idx in sorted(del_idx, key=int, reverse=True):
      del tracker1.frames[idx]
  tracker1.frames = sorted(list(set(tracker1.frames+tracker2.frames)), key=int)
  tracker1.pos = merge_2_dict(tracker1.pos, tracker2.pos)
  tracker1.features = merge_2_dict(tracker1.features, tracker2.features)

def merge_2_dict(x, y):
  z = x.copy()
  z.update(y)
  return z

# IO
parser = argparse.ArgumentParser()
parser.add_argument('tracking_csv', help='tracking result csv file')
parser.add_argument('video', help='location of corresponding video')
parser.add_argument('--temp_dir', default='./tmp', type=str, help='temp dir for saving img for reid')
parser.add_argument('--window', default=15, type=int, help='how many frames will tracker search to revive')
parser.add_argument('--th', default=200, type=int, help='feature distance threshold')
parser.add_argument('output_csv', help='output csv file')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--reid_model', required=True, type=str, help='reid cnn model')
parser.add_argument('--n_layers', type=int, required=True, help='# of layers of reid_model')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for reid cnn model')
opt = parser.parse_args()
os.system('mkdir -p %s' % opt.temp_dir)
os.system('rm -rf %s/*' % opt.temp_dir)

# Sort the detections by frame
detections = np.loadtxt(opt.tracking_csv, delimiter=',')
sort_idx = np.argsort(detections[:, 0])
detections = detections[sort_idx, :]

# Iterate through time, extract imgs to temp_dir for reid-CNN
print('Stage 1/3: Create tracker & extract img')
video = cv2.VideoCapture(opt.video)
if cv2.__version__[0] == '3':
   frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
else:
   frame_count = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
framenumber = 0
trackers = {}
trackers_sort_by_dead_time = []
trackers_hash_by_id = {}
pbar = ProgressBar(max_value=frame_count)
while True:
  ret, frame = video.read()
  if not ret: break
  framenumber += 1
  pbar.update(framenumber)
  
  frame[:, :, [0,2]] = frame[:, :, [2,0]]
  bboxs = detections[detections[:,0]==framenumber, :]
  bboxs = bboxs.reshape(-1, detections.shape[1])
  for i in range(bboxs.shape[0]):
    id_ = bboxs[i, 1]
    x0, y0 = (bboxs[i, 2:4]).astype('int')
    x1, y1 = (bboxs[i, 2:4] + bboxs[i, 4:6]).astype('int')
    cropped = frame[y0:y1, x0:x1]
    if id_ not in trackers:
      trackers[id_] = Tracker(id_, opt.temp_dir, opt.window)
    trackers[id_].update(cropped, [x0, y0, x1, y1], framenumber)

  dead_id = []
  for id_ in trackers.keys():
    die = trackers[id_].forward()
    if die: dead_id.append(id_)
  for id_ in dead_id:
    trackers_sort_by_dead_time.append(trackers[id_])
    trackers_hash_by_id[id_] = trackers_sort_by_dead_time[-1]
    del trackers[id_]
pbar.finish()

for id_ in trackers.keys():
  trackers[id_].forward()
  trackers_sort_by_dead_time.append(trackers[id_])
  trackers_hash_by_id[id_] = trackers_sort_by_dead_time[-1]

# Use reid-CNN to extract feature
print('Stage 2/3: Extract re-id feature')
imgs = [os.path.join(opt.temp_dir, img) for img in os.listdir(opt.temp_dir)]
reid_model = ResNet_Loader(opt.reid_model, n_layer=opt.n_layers, batch_size=opt.batch_size)
features = reid_model.inference(imgs)
features = features.numpy()
np.set_printoptions(threshold=100)
for idx, img in enumerate(os.listdir(opt.temp_dir)):
  fname = img.split('_')
  id_ = int(fname[0][:8])
  frame = int(fname[1][1:7])
  trackers_hash_by_id[id_].add_feature(features[idx], frame)
#for t in trackers_sort_by_dead_time:
#  t.print()

# Matching process:
# 1. check if any head-tail pair of different tracker are within time window
# 2. calculate IOU max & feature dist min
# 3. choose only 1 match that exceeds threshold condition
print('Stage 3/3: Matching')
if opt.verbose: print('============ Matching ===========')
main_idx = 0
while main_idx < len(trackers_sort_by_dead_time):
  main_tracker = trackers_sort_by_dead_time[main_idx]
  if opt.verbose: print('tracker died:', main_tracker.get_id())
  scores = []
  idxs = []
  for idx, comp_tracker in enumerate(trackers_sort_by_dead_time):
    if main_tracker.get_id() != comp_tracker.get_id() and \
       abs(main_tracker.get_tail() - comp_tracker.get_head()) < opt.window:
      # Match 2 trackers
      scores.append(match_trackers(main_tracker, comp_tracker))
      idxs.append(idx)
  if opt.verbose: print('* scores:', scores)
  if len(scores) != 0 and min(scores) < opt.th:
    min_idx = idxs[scores.index(min(scores))]
    # Merge 2 trackers
    merge_trackers(main_tracker, trackers_sort_by_dead_time[min_idx])
    if opt.verbose: print('===> tracker merged:', trackers_sort_by_dead_time[min_idx].get_id())
    del trackers_sort_by_dead_time[min_idx]
    if scores.index(min(scores)) > main_idx:
      main_idx += 1
  else:
    main_idx += 1
  if opt.verbose: print('------------------------------')
  #for t in trackers_sort_by_dead_time:
  #  t.print()
  
f = open(opt.output_csv, 'w')
for t in trackers_sort_by_dead_time:
  t.write_file(f)
f.close()

#os.system('rm -rf %s' % opt.temp_dir)






