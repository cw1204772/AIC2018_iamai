import pickle
import sys
import os
from PIL import Image
import numpy as np
import csv
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dir to BoxCars116k dataset')
    parser.add_argument('--image_dir', help='dir to BoxCars116k dataset(image)')
    parser.add_argument('--output_dir', help='dir to output txt')
    args = parser.parse_args()

    dataset_dir = str(pathlib.Path(args.dataset_dir).resolve())
    dataset_pkl = os.path.join(args.dataset_dir, 'dataset.pkl')
    split_pkl = os.path.join(args.dataset_dir, 'classification_splits.pkl')
    with open(dataset_pkl, 'rb') as f:
        dataset = pickle.load(f, encoding='latin-1')
    with open(split_pkl, 'rb') as f:
        split = pickle.load(f, encoding='latin-1')

    # Train
    outputs = [['img_path', 'id', 'x', 'y', 'w', 'h']]
    for vID, ID in (split['hard']['train']+split['hard']['validation']):
        instances = dataset['samples'][vID]['instances']
        for i in instances:
            img_path = os.path.join(args.image_dir, 'images', i['path'])
            x, y, w, h = i['2DBB']
            outputs.append([img_path, ID, x, y, w, h])
    with open(os.path.join(args.output_dir, 'BoxCars_train.txt'), 'w') as f:
         csv.writer(f, delimiter=' ').writerows(outputs)

    # Test
    outputs = [['img_path', 'id', 'x', 'y', 'w', 'h']]
    for vID, ID in (split['hard']['test']):
        instances = dataset['samples'][vID]['instances']
        for i in instances:
            img_path = os.path.join(args.image_dir, 'images', i['path'])
            x, y, w, h = i['2DBB']
            outputs.append([img_path, ID, x, y, w, h])
    with open(os.path.join(args.output_dir, 'BoxCars_test.txt'), 'w') as f:
         csv.writer(f, delimiter=' ').writerows(outputs)

    
