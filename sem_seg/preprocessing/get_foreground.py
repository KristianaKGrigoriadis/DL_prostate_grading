import argparse
import json
import numpy as np
import os

from PIL import Image

""" This file takes in an odgt file and returns an odgt file of all the images 
and labels that have foreground annotations present.

Flags:
    -o : path to .odgt file containing images and corresponding labels
"""

formatSpec = '"fpath_img": "{}", "fpath_segm": "{}", "width": {}, "height": {}'

def get_foreground(odgt):    
    base_dir = os.path.dirname(odgt) 
    odgtname = os.path.basename(odgt).split('.')[0]    
    list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
    list_file = open(os.path.join(base_dir, odgtname+'_foreground.odgt')  ,"w")
    
    for file in list_sample:
        labpath = os.path.join(file["fpath_segm"])
        label = Image.open(os.path.join(base_dir,labpath))
        arr = np.array(label).astype(np.int8)
        classes = np.unique(arr)
        if len(classes)==1:
            if classes.item() == 0:
                continue
        else:
            img_name = file["fpath_img"]
            lab_name = file["fpath_segm"]
            width = file["width"]
            height = file["height"]
            list_file.write('{' + formatSpec.format(img_name, lab_name, str(width), str(height)) + '}\n')

def main():
    parser = argparse.ArgumentParser(description='Get foreground images')
    parser.add_argument('-o', '--odgt', help='path to .odgt file', type=str)
    args = parser.parse_args()
    odgt = args.odgt
    
    get_foreground(odgt)
    print('done.')
    

if __name__=='__main__':
    main()

