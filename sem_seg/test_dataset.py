import json
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


class TestDataset(Dataset):
   
    def __init__(self, odgt, patch_size, device):
        self.parse_input_list(odgt)
        self.patch_size = patch_size
        self.device = device
    
    def parse_input_list(self, odgt):
        self.odgt_dir = os.path.dirname(odgt)
        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
        
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0, "Number of files should be > 0"
        print("Number of samples: {}".format(self.num_sample))
        
    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        def crop_image(image):
            list_crop_imgs = []
            coordinate_list = []
            
            overlap = 0
            count = 0
            
            ori_size = (image.size[0], image.size[1])
            print("\nori_size = {}".format(ori_size))


            x_iter_num = (image.size[0])//(self.patch_size-overlap)
            y_iter_num = (image.size[1])//(self.patch_size-overlap)
            print("\nx_iter_num ={}, y_iter_num ={}".format(x_iter_num,y_iter_num))

            for xi in range(x_iter_num+1):
                for yi in range(y_iter_num+1):
                    if xi == 0 or image.size[0] < self.patch_size:
                        cx = 0
                    elif xi == x_iter_num:
                        cx = image.size[0]-self.patch_size
                    else:
                        cx = xi*(self.patch_size-overlap)
                    
                    if yi == 0 or image.size[1] < self.patch_size:
                        cy = 0
                    elif yi == y_iter_num:
                        cy = image.size[1]-self.patch_size
                    else:
                        cy = yi*(self.patch_size-overlap)
                    
                    
                    if image.size[0] < self.patch_size:
                        patch_size_x_cur = image.size[0]
                    else:
                        patch_size_x_cur = self.patch_size
                    
                    if image.size[1] < self.patch_size:
                        patch_size_y_cur = image.size[1]
                    else:
                        patch_size_y_cur = self.patch_size

                    left = cx
                    upper = cy
                    
                    if cx + patch_size_x_cur > image.size[0]: 
                        right = image.size[0]
                    else:
                        right = cx + patch_size_x_cur
                        
                    if cy + patch_size_y_cur > image.size[1]:
                        lower = image.size[1]
                    else:
                        lower = cy + patch_size_y_cur
                    
                    box = (left, upper, right, lower)
                    
                    crop_ti = image.crop(box)                     

                    coordinate_list.append([cx, cy])
               
                    crop_tensor = np.array(crop_ti)[:,:,:3]
                    crop_tensor = torch.from_numpy(crop_tensor.transpose((2, 0, 1))).to(self.device)
                    # print(crop_tensor.shape)
                    crop_tensor = crop_tensor.type(torch.double)/255
                    crop_tensor = crop_tensor.type(torch.float)
                    crop_tensor = normalise(crop_tensor) 
                    crop_tensor = torch.unsqueeze(crop_tensor, 0)

                    list_crop_imgs.append(crop_tensor)
                    count +=1
            
            return list_crop_imgs, coordinate_list, ori_size
        
        # load image
        file_str = self.list_sample[index]
        im_path = os.path.join(self.odgt_dir, file_str["fpath_img"])
        lab_path = os.path.join(self.odgt_dir, file_str["fpath_segm"])
            
        image = Image.open(im_path)
        label = Image.open(lab_path)
                
        # convert label to tensor:
        label = np.array(label)
        label = torch.from_numpy(label)
        label = torch.unsqueeze(label,0)
        
        list_crop_imgs, coordinate_list, ori_size = crop_image(image)
        
        print('number of patches: ',len(list_crop_imgs))
        
        crop_outputs = dict()
        crop_outputs['patch_list'] = [x.contiguous() for x in list_crop_imgs]
        crop_outputs['name'] = file_str['fpath_img'].split('/')[1]

        return [crop_outputs, coordinate_list, ori_size], label
