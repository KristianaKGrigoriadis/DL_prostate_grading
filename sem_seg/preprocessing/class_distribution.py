import argparse
import json
import numpy as np
import os

from pathlib import Path
from PIL import Image

"""
Classes: 
    0. Black [0, 0, 0] (i.e. background/benign)
    1. White [255, 255, 255] (atrophy)
    2. Light blue [0, 255, 255] (Gleason Grade 4)
    3. Pink [255, 0, 255] (Benign)
    4. Yellow [255, 255, 0] (HGPIN)
    5. Dark blue [0, 135, 255] (Gleason Grade 3)
    
"""


def get_pix_distribution(directory):
    
    dirpath = Path(directory)
    dirgen = dirpath.glob('*.png')
    pix_dist = np.zeros(6)
    
    for i,file in enumerate(dirgen):
        labs = Image.open(str(file.absolute()))
        arr = np.array(labs).astype(np.int8)
        # print(i, "\t", np.unique(arr))
        for j in range(6):
            pix_dist[j] += np.sum(arr == j)
    
    print("\nPixel distribution among all images: ",pix_dist/np.sum(pix_dist))

def get_class_distribution(directory):
    
    dirpath = Path(directory)
    dirgen = dirpath.glob('*.png')
    class_dist = np.zeros(6)    
    for i,file in enumerate(dirgen):
        labs = Image.open(str(file.absolute()))
        arr = np.array(labs).astype(np.int8)
        cols = np.unique(arr)
        
        for j in cols:
            class_dist[j] += 1
    
    print("\nClass disrtibution among all images: ", class_dist)

def pix_distr_foregd(odgt):
    odgt_dir = os.path.dirname(odgt)
    list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
    pix_dist = np.zeros(6)
    for file_str in list_sample:
        lab_path = os.path.join(odgt_dir, file_str["fpath_segm"])
        labs = Image.open(lab_path)
        arr = np.array(labs).astype(np.int8)
        
        for j in range(6):
            pix_dist[j] += np.sum(arr == j)
    print("\nClass disrtibution among all images: ", pix_dist)
    
def main():
    parser = argparse.ArgumentParser(description='Get distribution of pixels')
    parser.add_argument('-d', '--directory', help='Label directory. Write as directory name followed by /', type=str)
    parser.add_argument('-o', '--odgt', help='.odgt path of images', type=str)
    args = parser.parse_args()
    odgt = args.odgt
    if args.directory:
        directory = args.directory
        get_pix_distribution(directory)
        get_class_distribution(directory)
    elif args.odgt:
        odgt = args.odgt
        pix_distr_foregd(odgt)
    print('Done.\n')
    
    
if __name__=='__main__':
    main()
