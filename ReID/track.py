import numpy as np

class Track(object):
    def __init__(self, dets):
        self.dets = dets
        self.id = dets[0, 1]
    def sample_dets(self, sample_interval):
        return self.dets[::sample_interval]
    def import_features(self, features):
        self.features = features
    def birth_time(self):
        if self.dets.shape[0] == 0: return -1
        else: return np.min(self.dets[:,0])
    def dead_time(self):
        if self.dets.shape[0] == 0: return int('inf')
        return np.max(self.dets[:,0])
    def sct_match(self, t, bbox_dist_th):
        pos1 = self.dets[-1,2:6]
        pos2 = t.dets[0,2:6]
        if bbox_dist(pos1, pos2) < bbox_dist_th:
            feat1 = self.features[-1,:]
            feat2 = t.features[0,:]
            return l2dist(feat1, feat2)
        else: return float('inf')
    def merge(self, t):
        t.dets = np.concatenate([self.dets, t.dets], axis=0)
        t.dets[:,1] = t.id
        t.features = np.concatenate([self.features, t.features], axis=0)
        self.dets = np.zeros((0,0))
    def summarized_feature(self):
        return np.mean(self.features, axis=0)
    def assign_seq_id(self, seq_id,loc_id):
        seq_ids = seq_id * np.ones((self.dets.shape[0],1))
        loc_ids = loc_id * np.ones((self.dets.shape[0],1))
        self.dets = np.concatenate([self.dets, seq_ids], axis=1)
        self.dets = np.concatenate([self.dets, loc_ids], axis=1)
    def dump(self):
        assert self.dets.shape[1] == 9
        return self.dets[:, [7]+list(range(7))]

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

