import numpy as np
from pathlib import Path
from PIL import Image

""" This takes in a single 3-channel (RGB) image and converts it to a 
2D numpy array with the following labels:
    0. Black [0, 0, 0] (i.e. background/benign)
    1. White [255, 255, 255]
    2. Light blue [0, 255, 255]
    3. Pink [255, 0, 255]
    4. Yellow [255, 255, 0]
    5. Dark blue [0, 135, 255]
    6. Dark green [0, 135, 0]
"""

col2lab = {
    "[0, 0, 0]" :0,
    "[255, 255, 255]" : 1,
    "[0, 255, 255]" : 2,
    "[255, 0, 255]" : 3,
    "[255, 255, 0]" : 4,
    "[0, 135, 255]" : 5,
    "[0, 135, 0]" : 6}


def colour2label(in_array):
    if in_array.shape[-1] ==4:
        in_array = in_array[:,:,:-1]
    
    output_array = np.zeros((in_array.shape[0],in_array.shape[1])) 
        
    for i in range(in_array.shape[0]):
        for j in range(in_array.shape[1]):
            pixel = in_array[i,j,:].tolist()
            
            if str(pixel) in col2lab.keys():    
                output_array[i,j] = col2lab[str(pixel)]

    return output_array

dirpath = '/home/kgrigori/Joe_annoted/Segmentation/'
mask_path = dirpath + 'masks/'
labels_path = dirpath + 'labels/'

def main():
    count=0
    for file in Path(mask_path).glob('*/*'):
        file_end = file.absolute().parent.name + '/' + file.name
        mask_im = np.array(Image.open(file))
        output_array = colour2label(mask_im)        
        np.save(labels_path + file_end, output_array)
        count+=1
        print(count)

if __name__ == "__main__":
    main()
