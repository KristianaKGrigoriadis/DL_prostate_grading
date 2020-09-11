import argparse
import numpy as np
import os

from pathlib import Path
from PIL import Image

""" Converts multi class labelled images to binary images - cancer or not cancer
"""

def multi_class2binary(array):
    bin_labels = (array > 0).astype(np.uint8)
    return bin_labels


def main():
    parser = argparse.ArgumentParser(description='Convert multi class labels to binary labels')
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    argu_list = [[it for it in item.split(': ')] for item in args.list.split(', ')]
    input_path = argu_list[0][1]
    output_path = argu_list[1][1]
        
    for label in Path(input_path).glob('*.png'):
        np_lab = np.array(Image.open(label.absolute())).astype(np.uint8)
        bin_labels = multi_class2binary(np_lab)
        bin_im = Image.fromarray(bin_labels, mode='L')
        bin_im.save(os.path.join(output_path,label.name))
    print("done")

if __name__ == '__main__':
    main()
    
