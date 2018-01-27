import sys
import argparse
import os
from scipy.io import loadmat
from PIL import Image
import csv

# Usage: python3 this.py <input_dataset_dir> <output_database_dir> <database_list.txt>
if __name__ == '__main__':

  # Argparse
  parser = argparse.ArgumentParser(description='Database generator for UA-DETRAC dataset')
  parser.add_argument('--img_dir', help='the dir containing dataset imgs')
  parser.add_argument('--label_dir', help='the dir containing dataset label files (MAT format)')
  parser.add_argument('--database_dir', help='the dir you wish to store database')
  parser.add_argument('--database_txt', help='the output txt file listing all imgs to database and its label')
  args = parser.parse_args()

  # List label file
  label_files = [os.path.join(args.label_dir, f) for f in os.listdir(args.label_dir)]

  # Check output dir
  os.system('mkdir -p %s' % args.database_dir)
  
  # Cut bbox
  id_ = 1
  txt = []
  imgs = 0
  for f in label_files:
    seq_name = os.path.splitext(os.path.basename(f))[0]
    print(seq_name)
    os.system('mkdir -p %s' % os.path.join(args.database_dir, seq_name))
    img_dir = os.path.join(args.img_dir, seq_name)
    label = loadmat(f)['gtInfo']
    x_center = label[0][0][0].astype('float')
    y_bottom = label[0][0][1].astype('float')
    h = label[0][0][2].astype('float')
    w = label[0][0][3].astype('float')
    frames = label[0][0][4].astype('int')
    for i in frames.squeeze().tolist():
      img = Image.open(os.path.join(img_dir, 'img%05d.jpg' % i))
      for j in range(w.shape[1]):
        if w[i-1][j] != 0 and h[i-1][j] != 0:
          x0 = x_center[i-1][j] - w[i-1][j]/2
          x1 = x_center[i-1][j] + w[i-1][j]/2
          y0 = y_bottom[i-1][j] - h[i-1][j]
          y1 = y_bottom[i-1][j]
          if x0 < 0: x0 = 0
          if x1 > img.size[0]: x1 = img.size[0]
          if y0 < 0: y0 = 0
          if y1 > img.size[1]: y1 = img.size[1]
          cropped = img.crop((x0, y0, x1, y1))
          save_img = os.path.join(args.database_dir, seq_name, '%04dF%05d.jpg' % (j, i))
          cropped.save(save_img)
          id = id_ + j
          txt.append([save_img, str(id)])        
          imgs += 1
    id_ += w.shape[1]

  # Write database_txt
  with open(args.database_txt, 'w') as outfile:
    writer = csv.writer(outfile, delimiter=' ')
    writer.writerows(txt)
  print('imgs:', imgs, 'id:', id_)
