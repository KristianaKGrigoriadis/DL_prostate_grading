import argparse
import numpy as np
import os

from pathlib import Path
from PIL import Image

""" This file generates image patches with corresponding the label image. 
It also writes .odgt files, where each line is one patch: 
    {"fpath_img": , 
    "fpath_segm": , 
    "width": , 
    "height": }     
It should be run in the following way:
    python patch_preparation.py -l 'base_path: \BASE_PATH, start_patch_size: 500, end_patch_size: 600, step_size: 50, overlap: 200, \
    foreground: 0.2, intensity_threshold: 195'
"""


def crop_image(base_path, patch_size, overlap, foreground, intensity_threshold):
    # note: base_path: /home/kgrigori/patch_data in cluster
    img_path = base_path + '/images/'
    lab_path = base_path + '/labels/'    
    base_path_crop = os.path.join(base_path, str(patch_size))
    crop_img_path = base_path_crop + '/images'
    crop_lab_path = base_path_crop + '/labels'

    if not os.path.exists(base_path_crop):
        os.mkdir(base_path_crop)
        os.mkdir(crop_img_path)
        os.mkdir(crop_lab_path)

    formatSpec = '"fpath_img": "images/{}", "fpath_segm": "labels/{}", "width": {}, "height": {}';
    count = 0

    if len(os.listdir(crop_img_path)) < len(os.listdir(img_path)):
        print('\nCropping data\n')
        list_file = open(base_path_crop + "/all_patches.odgt","w")

        for imgs in Path(img_path).glob('*/*.png'):
            image = Image.open(str(imgs.absolute()))
            currlab_path = next(Path(lab_path).glob('*/'+imgs.name))
            lab_img = Image.open(currlab_path)
            img_grade = imgs.parent.name
            
            x_iter_num = (image.size[0])//(patch_size-overlap)
            y_iter_num = (image.size[1])//(patch_size-overlap)
            print("Cropping patches for ", imgs, 'image size: ', image.size, 'x_iter_num=',x_iter_num,'y_iter_num=',y_iter_num,'\n')
            patch_count=0
            for xi in range(x_iter_num+1):
                for yi in range(y_iter_num+1):
                    if xi == 0 or image.size[0] < patch_size:
                        cx = 0
                    elif xi == x_iter_num:
                        cx = image.size[0]-patch_size
                    else:
                        cx = xi*(patch_size-overlap)
                    if yi == 0 or image.size[1] < patch_size:
                        cy = 0
                    elif yi == y_iter_num:
                        cy = image.size[1]-patch_size
                    else:
                        cy = yi*(patch_size-overlap)
                    if image.size[0] < patch_size:
                        patch_size_x_cur = image.size[0]
                    else:
                        patch_size_x_cur = patch_size
                    if image.size[1] < patch_size:
                        patch_size_y_cur = image.size[1]
                    else:
                        patch_size_y_cur = patch_size
                    
                    # don't include a patch if it crosses the image border
                    if cx + patch_size_x_cur > image.size[0]:
                        continue
                    if cy + patch_size_y_cur > image.size[1]:
                        continue
                    else:
                        box = (cx, cy, cx + patch_size_x_cur, cy + patch_size_y_cur)
                        new_name_ti = 'image_' + imgs.name + '_grade_' + img_grade + '_cropped_' + str(patch_size) + '_x_' + str(cx) + '_y_' + str(cy) + '.png'
                        new_name_gt = 'label_' + imgs.name + '_grade_' + img_grade + '_cropped_' + str(patch_size) + '_x_' + str(cx) + '_y_' + str(cy) + '.png'
                        crop_ti = image.crop(box)
                        
                        # check to see if this image has sufficient foreground:
                        np_crop_ti = np.array(crop_ti.convert('L'))
                        total_px_crop = np_crop_ti.shape[0]*np_crop_ti.shape[1]
                        foreground_crop_im = np_crop_ti < intensity_threshold
                        prop = np.sum(foreground_crop_im)/total_px_crop
                        if prop < foreground:
                            continue
                        else:
                            # print('xi = {}, yi = {}, cx = {}, cy = {}'.format(xi,yi,cx,cy))
                            save_path_ti = os.path.join(crop_img_path, new_name_ti)
                            crop_ti.save(save_path_ti)
                            crop_gt = lab_img.crop(box)
                            save_path_gt = os.path.join(crop_lab_path, new_name_gt)
                            crop_gt.save(save_path_gt)
        
                            list_file.write('{' + formatSpec.format(new_name_ti, new_name_gt, str(patch_size), str(patch_size)) + '}\n')
                            patch_count +=1
            count += 1
        print('{} images cropped, {} patches generated'.format(count,patch_count))
        list_file.close()

def main():
    parser = argparse.ArgumentParser(description='Crop Patches')
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    argu_list = [[it for it in item.split(': ')] for item in args.list.split(', ')]
    base_path = argu_list[0][1]
    start_patch_size = int(argu_list[1][1])
    end_patch_size = int(argu_list[2][1])
    step_size = int(argu_list[3][1])
    overlap = int(argu_list[4][1])
    foreground = np.float16(argu_list[5][1])
    intensity_threshold = int(argu_list[6][1])
    cur_patch_size = start_patch_size
    while cur_patch_size <= end_patch_size:
        crop_image(base_path, cur_patch_size, overlap, foreground, intensity_threshold)
        cur_patch_size += step_size

if __name__ == '__main__':
    main()

