import argparse
import numpy as np

from loss_functions import intersectionAndUnion
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Get IoU of predictions')
    parser.add_argument('-p', '--predictions',type=str)
    parser.add_argument('-l', '--labels',type=str)
    parser.add_argument('-n', '--num_class',type=int)
    parser.add_argument('-i', '--ignore_index',type=int)
    args = parser.parse_args()
    
    num_class = args.num_class
    sum_intersection = np.zeros(num_class)
    sum_union = np.zeros(num_class)
    
    for pred in Path(args.predictions).glob('*.png'):
        print(pred.name)
        label = next(Path(args.labels).glob(pred.name))
        
        imPred = Image.open(str(pred.absolute()))
        imPred = np.array(imPred).astype(np.uint8)
        
        imLab = Image.open(str(label.absolute()))
        imLab = np.array(imLab).astype(np.uint8)
        if args.ignore_index:
            intersection, union,_ = intersectionAndUnion(imPred,imLab,num_class,args.ignore_index)
        else:
            intersection, union,_ = intersectionAndUnion(imPred,imLab,num_class)            
        sum_intersection += intersection
        sum_union += union
    
    IoU = sum_intersection/sum_union
    mIoU = IoU.mean()
    
    print('IoU =',IoU)
    print('mIoU =',mIoU)

if __name__=="__main__":
    main()
