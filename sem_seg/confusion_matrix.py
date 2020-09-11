import argparse
import numpy as np
import os
import pandas as pd

from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix

def get_CM(pred_dir, lab_dir):
    pred_array = np.array([])
    lab_array = np.array([])
    
    for pred in Path(pred_dir).glob('*.png'):
        name = pred.name
        print(name)
        label = next(Path(lab_dir).glob(name))
        labels = Image.open(label.absolute())    
        labels = np.array(labels).astype(np.uint8)
        labels = labels.ravel()
        lab_array = np.concatenate((lab_array,labels))

        preds = Image.open(pred.absolute())
        preds = np.array(preds).astype(np.uint8)                        
        preds = preds.ravel()
        pred_array = np.concatenate((pred_array, preds))
        
    cm = confusion_matrix(lab_array, pred_array)
    return cm

def get_CM_big(pred_dir,lab_dir,num_class):
    
    big_CM = np.zeros((num_class,num_class))
    for pred in Path(pred_dir).glob('*.png'):
        name = pred.name
        print(name)
        label = next(Path(lab_dir).glob(name))
        labs = Image.open(label.absolute())    
        labs = np.array(labs).astype(np.uint8)
        labs = labs.ravel()
        
        preds = Image.open(pred.absolute())
        preds = np.array(preds).astype(np.uint8)                        
        preds = preds.ravel()
        
        cm = confusion_matrix(labs, preds,labels=list(range(num_class)))
        
        big_CM += cm
    
    return big_CM

def main():
    parser = argparse.ArgumentParser(description='Get confusion matrix')
    parser.add_argument('-p','--pred_dir',type=str)
    parser.add_argument('-l','--lab_dir',type=str)
    parser.add_argument('-n','--num_class',type=int)
    args = parser.parse_args()
    
    cm = get_CM_big(args.pred_dir, args.lab_dir, args.num_class)
    df = pd.DataFrame(data=cm)
    df.to_csv(os.path.join(args.pred_dir,'confusion_matrix.csv'))

if __name__=="__main__":
    main()

