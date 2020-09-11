import argparse
import numpy as np
from pathlib import Path
from PIL import Image

formatSpec = '"fpath_img": "images/{}", "fpath_segm": "labels/{}", "width": {}, "height": {}';

def split(directory):
    
    patch_size = directory.split('/')[-2]
    
    images_path = directory + 'images/'
    labels_path = directory + 'labels/'
    
    images_list = list(Path(images_path).glob('*'))

    range_arr = np.arange(len(images_list))
    np.random.shuffle(range_arr)
    print("The total number of images are: ", len(range_arr))
    
    split_dir = directory + 'split/'
    
    train_path = split_dir + 'train/'
    val_path = split_dir + 'val/'
    test_path = split_dir + 'test/'
    
    im_train_path = train_path + 'images/'
    im_val_path = val_path + 'images/'    
    im_test_path = test_path + 'images/'
    
    lab_train_path = train_path + 'labels/'
    lab_val_path = val_path + 'labels/'    
    lab_test_path = test_path + 'labels/'
    
    train_odgt_file = open(train_path + patch_size + "_train_patches.odgt","w")
    val_odgt_file = open(val_path + patch_size + "_val_patches.odgt","w")
    test_odgt_file = open(test_path + patch_size + "_test_patches.odgt","w")    
    
    for i, index in enumerate(range_arr):
        im_file = images_list[index].absolute()
        raw_im = Image.open(str(im_file))
        
        lab_file = labels_path + im_file.name
        lab_im = Image.open(lab_file)
        
        if i < (0.75 * len(range_arr)):
            # save images to train subdir
            raw_im.save(im_train_path + im_file.name)
            lab_im.save(lab_train_path + im_file.name)
            # write to .odgt file
            train_odgt_file.write('{' + formatSpec.format(im_file.name, im_file.name, patch_size, patch_size) + '}\n')
            
        elif (0.75 * len(range_arr)) <= i <  (0.85 * len(range_arr)):
            # save images to val subdir
            raw_im.save(im_val_path + im_file.name)
            lab_im.save(lab_val_path + im_file.name)
            # write to .odgt file
            val_odgt_file.write('{' + formatSpec.format(im_file.name, im_file.name, patch_size, patch_size) + '}\n')
            
        elif (0.85 * len(range_arr)) < i:
            # save images to test subdir
            raw_im.save(im_test_path + im_file.name)
            lab_im.save(lab_test_path + im_file.name)
            # write to .odgt file
            test_odgt_file.write('{' + formatSpec.format(im_file.name, im_file.name, patch_size, patch_size) + '}\n')
    print('Done.')
            
def main():
    print('Running main()')
    parser = argparse.ArgumentParser(description='Split into train/val/test')
    parser.add_argument('-d', '--directory', help='Write as directory name followed by /', type=str)
    args = parser.parse_args()
    print(args.directory, type(args.directory))
    curr_dir = args.directory
    split(curr_dir)

if __name__ == '__main__':
    main() 

