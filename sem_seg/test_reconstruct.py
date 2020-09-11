import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import utils

from collections import OrderedDict
from PIL import Image
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from unet import UNet


lab2col = {
    '0' : [0, 0, 0],
    '1' : [255, 255, 255],
    '2' : [0, 255, 255],
    '3' : [255, 0, 255],
    '4' : [255, 255, 0],
    '5' : [0, 135, 255]}

lab2col_np = np.zeros((6,3))
for lab in lab2col.keys():
    lab2col_np[int(lab),:]=lab2col[lab]
lab2col_np = lab2col_np.astype(np.uint8)

def label2colour(np_array):
    height, width= np_array.shape
    if len(np.unique(np_array))>2:
        outmask = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                lab = np_array[i,j]
                # print(lab)
                outmask[i,j,:] = lab2col_np[lab,:].astype(np.uint8)
        print('outmask.shape =',outmask.shape)
        outmask = Image.fromarray(outmask.astype(np.uint8),mode='RGB')
    else:
        outmask = label2gray(np_array)
    return outmask

def label2gray(np_array):
    np_array = np_array.astype(np.uint8)
    outmask = Image.fromarray(255*np_array,mode='L')  
    return outmask


def reconstruct(crop_pred_list, coordinate_list, ori_size, num_class):
    pred = torch.zeros((1,num_class,ori_size[1],ori_size[0]))
    print("pred.shape = {}".format(pred.shape))
    for idx in range(len(crop_pred_list)):
        cx, cy = coordinate_list[idx]
        _, num_channels, size_y, size_x = crop_pred_list[idx].shape
        pred[:,:, cy:cy+size_y, cx:cx+size_x] = crop_pred_list[idx]
    return pred



def test(model, dataloader, num_class, device, out_path):
    model.eval()
    num_class = model.out_ch
    
    test_IoU = torch.zeros(num_class)
    test_Dice = torch.zeros(num_class)    
    test_accuracy = 0
    
    for i, (batch_ims,batch_labs) in enumerate(dataloader):
        print('\nBatch:',i)
        for j,image in enumerate(batch_ims):
            crop_outputs, coordinate_list, ori_size = image
            
            crop_pred_list = []
            
            patch_list = crop_outputs['patch_list']
            image_name = crop_outputs['name']
            print('Image name:',image_name)
            
            with torch.no_grad():                
                for impatch in patch_list:
                    patch_input = impatch.clone().float()
                    patch_input = patch_input.to(device)
                    pred_tmp = model(patch_input)
                    
                    crop_pred_list.append(pred_tmp)
            
            # now generate a probability array for one singular WSI
            probs = reconstruct(crop_pred_list, coordinate_list, ori_size, num_class)
            
            # make prediction by taking argmax
            labels = batch_labs[j]
            labels = labels.float()
            # print('probs.shape =',probs.shape)
            preds = probs.argmax(1).float()
            print("unique predictions: ",np.unique(preds.cpu().numpy()))
            
            # evaluate accuracy:
            correct = (preds == labels).sum().item()
            total = (labels==labels).sum().item()
            # print('correct = {}, total = {}'.format(correct,total))
            acc = correct/total
            test_accuracy += acc
            
            # evaluate IoU of this prediction:
            labels = labels.type(torch.LongTensor)
            target_1_hot = torch.eye(num_class)[labels.squeeze(1)]
            target_1_hot = target_1_hot.type(probs.type())            
            # target_1_hot = torch.transpose(target_1_hot,1,3)
            # target_1_hot = torch.transpose(target_1_hot,2,3)
            target_1_hot = target_1_hot.permute(0, 3, 1, 2)
            dims = (0,) + tuple(range(2, probs.ndimension()))
            intersection = torch.sum(probs * target_1_hot, dims)
            cardinality = torch.sum(probs + target_1_hot, dims)
            union = cardinality - intersection
            jacc_index = (intersection / (union + 1e-7))
            # add this IoU to variable test_IoU
            test_IoU += jacc_index
            
            # evaluate Dice coefficient:
            dice = (2. * intersection / (cardinality + 1e-7))            
            print('IoU = {}, accuracy = {}'.format(jacc_index,acc))
            test_Dice += dice

            prediction = np.array(preds).astype(np.uint8)
            prediction = prediction.squeeze(0)
            # print('prediction.shape = {}'.format(prediction.shape))

            # save prediction to file
            pred_im = Image.fromarray(prediction, mode = 'L') 
            pred_im.save(os.path.join(out_path, 'preds',image_name))            
            
            # save mask to file
            # print("num_class = ",num_class)
            if num_class==2:
                outmask = label2gray(prediction)
            else:
                outmask = label2colour(prediction)

            outmask.save(os.path.join(out_path, 'masks',image_name))
    
    test_IoU = test_IoU/len(dataloader.dataset)
    test_Dice = test_Dice/len(dataloader.dataset)    
    test_accuracy = test_accuracy/len(dataloader.dataset)

    return test_IoU, test_Dice, test_accuracy
    


def main():
    
    parser = argparse.ArgumentParser(description='Test on WSIs')    
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    argu_list = [[it for it in item.split(': ')] for item in args.list.split(', ')]
    odgt_path = argu_list[0][1]
    patch_size = int(argu_list[1][1])
    num_class = int(argu_list[2][1])
    batch_size = int(argu_list[3][1])
    modelpath = argu_list[4][1]
    out_path = argu_list[5][1]    
    
    # initialising the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice: ', device)
    
    print('odgt_path =',odgt_path)
    
    test_ds = TestDataset(odgt_path,patch_size, device)
    
    # loader for test set 
    test_dl = DataLoader(dataset = test_ds,
                         batch_size = batch_size,
                         shuffle = False,
                         collate_fn = utils.collate_test)
    
    model = UNet(out_ch=num_class)       
    checkpoint = torch.load(modelpath)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k[:7]=='module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

        
    test_IoU, test_Dice, test_accuracy = test(model, test_dl, num_class, device, out_path)

    print("\nIoU = {}, Dice = {}, accuracy = {:3f}%".format(test_IoU, test_Dice, test_accuracy*100))

    
    
if __name__=="__main__":
    main()
    
