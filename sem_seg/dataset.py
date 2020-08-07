import json
import numpy as np
import os
import torch 

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


""" 
Differences between train and val sets:
    - Training data is augmented, val and test data is not
"""


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


# Data augmentation and normalization for training
# Just normalization for validation and test

class RotationTransform:
    """
    Rotate by one of the given angles.
        Args:
            - angles: a list containing angles to choose from 
        When called:
            - x: an (image, segmentation) tuple
    """
    def __init__(self, angles):
        self.angles = np.array(angles)

    def __call__(self, x):
        image, segmentation = x
        angle = np.random.choice(self.angles)
        rotated_im = image.rotate(angle)
        rotated_seg = segmentation.rotate(angle)
        return rotated_im, rotated_seg


class HorizontalFlip:
    """ 
    Randomly flip horizontally with probability p.
        Args:
            - p: probability with which to flip the patch (default=0.5),
        When called:
            - x: an (image, segmentation) tuple
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        image, segmentation = x
        flip = np.random.choice([0,1],p=[1-self.prob, self.prob])
        
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            segmentation = segmentation.transpose(Image.FLIP_LEFT_RIGHT)
        return image, segmentation

class VerticalFlip:
    """ 
    Randomly flip vertically with probability p.
        Args:
            - p: probability with which to flip the patch,
        When called:
            - x: an (image, segmentation) tuple
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, x):
        image, segmentation = x
        flip = np.random.choice([0,1],p=[1-self.prob, self.prob])
        
        if flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            segmentation = segmentation.transpose(Image.FLIP_TOP_BOTTOM)
        return image, segmentation

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, x):
        image, segmentation = x
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image)[:,:,:3]
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.type(torch.double)/255
        
        label = np.array(segmentation)
        label = torch.from_numpy(label)
        return image, label

class NormalizeTransform:
    """ 
    Normalises the input image using transforms.Normalize() 
    Args:
        - mean : [R,G,B] default,
        - std: [R,G,B] default
    """
    def __init__(self, mean, std):
        self.normalise = transforms.Normalize(mean, std)
    def __call__(self, x):
        image, segmentation = x
        im_norm = self.normalise(image)      
        return im_norm, segmentation


data_transform = {
    'train': transforms.Compose(
        [RotationTransform(angles= [90,180,270]),
          VerticalFlip(),
          HorizontalFlip(),
          ToTensor(),
          NormalizeTransform([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    'test': transforms.Compose(
        [ToTensor(),
         NormalizeTransform([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}


class SegmentationDataset(Dataset):
    """ Patch-based segmentation dataset """
    def __init__(self, odgt, transform = data_transform, train=True):
        """ 
        Args:
            odgt: a path to the .odgt file containing the paths to patches to create the dataset from,
            train: whether this data is training data,
            transform: the dictionary of transforms we can apply to the dataset
            """
        
        self.parse_input_list(odgt)
        self.patch_width = self.list_sample[0]["width"] # for now I am uploading patches with all the same width and height
        self.patch_height = self.list_sample[0]["height"]
                
        self.train = train
        if self.train:
            self.transform = data_transform['train']
        else:
            self.transform = data_transform['test']

    def parse_input_list(self, odgt):
        self.odgt_dir = os.path.dirname(odgt)
        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
        
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0, "Number of files should be > 0"
        print("Number of samples: {}".format(self.num_sample))
    
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self,index, interp='bilinear'):
        # print("index =",index)
        file_str = self.list_sample[index]
        im_path = os.path.join(self.odgt_dir, file_str["fpath_img"])
        lab_path = os.path.join(self.odgt_dir, file_str["fpath_segm"])
            
        im = Image.open(im_path)
        lab = Image.open(lab_path)

        if im.size[0] !=self.patch_height or im.size[1] != self.patch_width:
            im  = imresize(im, (512,512), interp)
            lab  = imresize(lab, (512,512), interp)

        transformed_im, transformed_lab = self.transform((im, lab))
        sample = (transformed_im,transformed_lab)
        
        return sample
        
