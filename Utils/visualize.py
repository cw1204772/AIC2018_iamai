import cv2
import sys
import numpy as np
import progressbar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('INPUT_VIDEO', help='Input video')
parser.add_argument('OUTPUT_VIDEO', help='Output video')
parser.add_argument('LABEL_FILE', help='Label file')
parser.add_argument('MODE', help="CONF or ID. If CONF, visualize detection's confidence score. If ID, visualize person ID.")
parser.add_argument('--w', default=640, help='Output video width. (Default: 640)')
parser.add_argument('--h', default=360, help='Output video height. (Default: 360)')
parser.add_argument('--fps', default=60, help='Output video fps. (Default: 60)')
parser.add_argument('--length', default=-1, help='Length of output video(frames). If not specified, it will be same as input video.')
parser.add_argument('--delimiter', default=' ', help='Delimiter of the label file.')
parser.add_argument('--offset', default=0, help="Offset between video's and label's frame number. If 0, video's frame number equals video's frame number. If 1, video's frame 1 is label's frame 2. If -1, video's frame 2 is label's frame 1. (Default: 0)")
parser.add_argument('--frame_pos', default=2, help="Frame number's column index(0-based). (Default: 2)")
parser.add_argument('--bbox_pos', default=3, help="Bounding box's column index(0-based). If 3, column 3,4,5,6 represents x,y,w,h of a bounding box. (Default: 3)")
parser.add_argument('--id_pos', default=1, help="[ID MODE only] Person ID's column index(0-based). (Default: 1)")
parser.add_argument('--score_pos', default=6, help="[CONF MODE only] Detection confidence score's column index(0-based). (Default: 6)")
parser.add_argument('--score_th', default=0, help="[CONF MODE only] Detection's confidence score threshold. (Default: 0)")
parser.add_argument('--cam', default=-1, help="If specified, filter labels according to cam.")
parser.add_argument('--cam_pos', default=0, help="Cam's column index. Only used when --cam.")
parser.add_argument('--ss', default=1, help="If specified. Bounding boxes in label file with frame >= ss & frame <= ss+length-1 are selected to visualize.")
parser.add_argument('--wh_mode', default=False, action='store_true', help='Whether bbox is represent with (x,y,w,h). If not specified, bbox is expected to be represented by (x0,y0,x1,y1)')
args = parser.parse_args()

# IO
cam_pos   = int(args.cam_pos)
frame_pos = int(args.frame_pos)
id_pos    = int(args.id_pos)
bbox_pos  = int(args.bbox_pos)
score_pos = int(args.score_pos)
start_frame = int(args.ss)
score_th = float(args.score_th)
offset = int(args.offset)
wh_mode = args.wh_mode
if args.MODE == 'CONF':
  mode = 'conf'
elif args.MODE == 'ID':
  mode = 'id'
else:
  sys.exit('Unknown mode!')

# Read input video
in_video = cv2.VideoCapture(args.INPUT_VIDEO)
if cv2.__version__[0] == '3':
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
   h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
   frame_count = in_video.get(cv2.CAP_PROP_FRAME_COUNT)
else:
   fourcc = cv2.cv.CV_FOURCC(*'XVID')
   w = in_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
   h = in_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
   frame_count = in_video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

# Create output video
fps = args.fps
w = args.w
h = args.h
out_video = cv2.VideoWriter(args.OUTPUT_VIDEO, fourcc, float(fps), (int(w),int(h)))

# Load label file
label = np.loadtxt(args.LABEL_FILE, delimiter=args.delimiter)


# Label preprocessing
# 1. Filter by cam
if args.cam != -1:
  label = label[label[:, cam_pos] == cam,:]
# 2. Filter by start frame
label = label[(label[:, frame_pos] >= start_frame), :]
# 3. Filter by detection score
if args.MODE == 'CONF': 
  label = label[label[:, score_pos] >= score_th, :]
# 4. Offset the label to match video
label[:, frame_pos] = label[:, frame_pos] - offset

# Color palette & font
color = [[0,0,255],
[0,255,0],
[255,0,0],
[0,255,255],
[255,255,0],
[255,0,255],
[0,128,255],
[0,128,0],
[255,128,0],
[128,0,255],
[255,0,128]
]
'''
[[180,119,31],
[14,127,255],
[44,160,44],
[40,39,214],
[189,103,148],
[75,86,140],
[194,119,227],
[127,127,127],
[34,189,188],
[207,190,23],
[51,255,255]
]

[[234,45,104],
[225,53,43],
[226,91,30],
[236,162,31],
[223,213,50],
[173,228,59],
[92,209,75],
[93,82,223],
[97,33,167],
[181,93,221],
[202,62,216],
[225,54,167]]
'''
n_colors = len(color)
rect_thickness = 3
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2

# Labeling
unique_frame = np.unique(label[:, frame_pos])
length = frame_count if args.length==-1 else int(args.length)
framenumber = 1
unique_counter = 0
bar = progressbar.ProgressBar(max_value=frame_count if length==-1 else length)
while(True):
   ret, frame = in_video.read()
   if not ret: break
   if framenumber > length: break
   if unique_counter < len(unique_frame) and framenumber == unique_frame[unique_counter]:
      entries = label[label[:, frame_pos] == framenumber,:]
      for i in range(entries.shape[0]):
         x0 = int(entries[i, bbox_pos])
         y0 = int(entries[i, bbox_pos+1])
         x1 = int(entries[i, bbox_pos+2])
         y1 = int(entries[i, bbox_pos+3])
         if wh_mode:
           x1 = x0 + x1
           y1 = y0 + y1
         if mode == 'id':
           text = int(entries[i, id_pos])
           c = (color[text % n_colors])
         elif mode == 'conf':
           text = '%.1f' % entries[i, score_pos]
           c = (color[0])
         cv2.rectangle(frame, (x0, y0), (x1, y1), c, rect_thickness)
         cv2.putText(frame, str(text), (x0, y0-3), font, font_scale, c, font_thickness)
      unique_counter += 1
   frame = cv2.resize(frame, (int(w),int(h)))
   out_video.write(frame)
   bar.update(framenumber-1)
   framenumber += 1
   
bar.finish()
   
in_video.release()
out_video.release()
