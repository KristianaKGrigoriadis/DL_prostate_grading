import numpy as np
import time

from PIL import Image
from pathlib import Path

print('\n Done importing!\n')

start = time.time()

""" 
This file splits our data into input images and masks and writes them to a 
new directory:

We create a 3D "mask" image given a raw image and the corresponding 
annotated image. As black is one of the colours used to annotate the slides, 
the annotations in the masks are taken as (255 - [R,G,B]) and the mask 
background is black (this means we are able to distinguish between the black 
annotations - now white - and the background).

Examples are only included if they have annotations.

There is also a 2D .npy file written of the mask with labels for each colour.

"""

dirpath = '/home/kgrigori/Joe_annoted/Segmentation/'
raw_path = Path(dirpath+'Raw/')
GT_path = Path(dirpath+'GT/')

images_path = dirpath + 'images/'
masks_path = dirpath + 'masks/'
labels_path = dirpath + 'labels/'

exts = ['.png', '_start.png']

count=0

for file in raw_path.glob('*/*.png'):
    filename = file.name[:-4]
    if filename[0] != '.':
        raw_im = Image.open(str(file.absolute()))
        raw_im_np = np.asarray(raw_im)
    
        for e in exts:
            temp = list(GT_path.glob('*/'+filename + e))
            if len(temp)>0:
                GT_file = temp[0]
        
        GT_im = Image.open(str(GT_file.absolute()))
        GT_im_np = np.asarray(GT_im)
        
        if GT_im_np.shape[-1]==4:
            GT_im_np = GT_im_np[:,:,:-1]
        if raw_im_np.shape[-1]==4:
            raw_im_np = raw_im_np[:,:,:-1]
            raw_im = Image.fromarray(raw_im_np)
        
        if GT_im_np.shape == raw_im_np.shape:
            diff = (GT_im_np - raw_im_np)!=0
            
            if np.any(diff)==True:
                file_end = file.absolute().parent.name + '/' + filename + '.png'
                print(file_end)
                                
                diff_sum = np.sum(diff, axis=2)
                mask_outline = (diff_sum>0)
                mask_outline = np.array((mask_outline,mask_outline,mask_outline)).transpose(1,2,0)
                
                annotations = GT_im_np * mask_outline
                background = 255*(1 - mask_outline).astype('uint8')
                
                mask_out = annotations + background
                mask_out = (255 - mask_out).astype('uint8')
                mask = Image.fromarray(mask_out)
                
                # write image and mask to new files
                

                raw_im.save(images_path + file_end)
                mask.save(masks_path +  file_end)
                
                count +=1
                print(count)

print('Script took {:.3f} seconds to run'.format(time.time()-start))


