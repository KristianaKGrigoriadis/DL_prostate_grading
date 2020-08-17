import argparse
import loss_functions
import numpy as np
import segmentation_dataset
import time
import torch
import torch.nn as nn
import utils

from torch.utils.data import DataLoader
# from torchsummary import summary
from unet import UNet


""" To run:
python3 train2.py -l 'dirpath: /home/kgrigori/patch_data/512/, learningrate: 0.01, batch_size: 1, max_epochs: 100, channels_out: 6, loss: dice'"""

# define a function to train the model
def train_fn(train_data, model, criterion, optimizer, device):
    print('Training one epoch.')
    # save the performance on training data
    train_loss = 0
    train_acc = 0
    for i, (images, labels) in enumerate(train_data):
        # print('Batch number {}'.format(i))
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        train_loss +=loss.item()
        loss.backward()
        optimizer.step()
        train_acc += utils.accuracy(output, labels)
        #print('Batch number {}, \t loss = {}'.format(i, loss.item()))
    return train_loss/len(train_data) , train_acc / len(train_data.dataset)

# define a function to evaluate model performance
def test_fn(test_data, model, criterion, device):
    print('Evaluating model on validation data')    
    test_loss = 0
    test_acc = 0
    for i, (images,labels) in enumerate(test_data):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
            loss = criterion(output,labels)
            test_loss += loss.item()
            test_acc += utils.accuracy(output, labels)
    
    return test_loss/len(test_data) , test_acc/len(test_data.dataset)


def run_training(train_dl, val_dl, model, loss, optimizer, 
                 batch_size, patch_size, curtime, device, max_epochs, save=True):
    print('\nTraining the model.')
    
    modelsave = './patch_size_'+ str(patch_size) + '_batch_size_'+str(batch_size)+'_best_model.tar'
    
    losses_train = []
    accs_train = []
    losses_val = []
    accs_val = []
    
    least_val_loss = 1e6
    best_val_acc = 0
    
    start_time = time.time()
    for epoch in range(max_epochs): 
        print('Epoch: %d' %(epoch))

        train_loss, train_acc = train_fn(train_dl, model, loss, optimizer, device)
        val_loss, val_acc = test_fn(val_dl, model, loss)
        
        losses_train.append(train_loss)
        accs_train.append(train_acc)
        losses_val.append(val_loss)
        accs_val.append(val_acc)
        
        # update best validation accuracy and lowest validation loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            if save==True:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_accuracy': best_val_acc,
                            'batch_size': batch_size,
                            'best_learning_rate': optimizer.defaults['lr']
                            },modelsave)
        
    
        # if epoch % 10 == 0:
        print(f'\t Train: \tLoss: {train_loss:.6f}\t|\tAcc: {train_acc * 100:.3f}%(train)')
        print(f'\t Validation: \tLoss: {val_loss:.6f}\t|\tAcc: {val_acc * 100:.3f}%(valid)','\n')
                
        trainfile = open('./training_check_file_'+patch_size+'_' + curtime,'a')
        trainfile.write('\nFinished training epoch '+str(epoch)+'\n')
        trainfile.write(f'\t Train: \tLoss: {train_loss:.6f}\t|\tAcc: {train_acc * 100:.3f}%(train)\n')
        trainfile.write(f'\t Validation: \tLoss: {val_loss:.6f}\t|\tAcc: {val_acc * 100:.3f}%(valid)\n')
        trainfile.close()

        # convergence criterion
        if epoch >= 50 and all((val_loss - torch.tensor(losses_val[-20:-1])) > 0.0):
            break
        
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    print("Training time : %d minutes, %d seconds" %(mins, secs))
    
    # print best loss and acc achieved
    ltrain = np.array(losses_train)  
    least_train_epoch = np.argmin(ltrain)
    least_train_loss = min(losses_train)
    print('Lowest training loss:',least_train_loss,'achieved at epoch:',least_train_epoch)
    
    lval = np.array(losses_val)  
    least_val_epoch = np.argmin(lval)
    least_val_loss = min(losses_val)
    print('Lowest validation loss:',least_val_loss,'achieved at epoch:',least_val_epoch)
    
    atrain = np.array(accs_train)  
    best_train_epoch = np.argmax(atrain)
    best_train_accuracy = max(accs_train)
    print('Best training accuracy:',best_train_accuracy * 100,'achieved at epoch:',best_train_epoch)
    aval = np.array(accs_val)  
    best_val_epoch = np.argmax(aval)
    best_val_accuracy = max(accs_val)
    print('Best validation accuracy:',best_val_accuracy*100,'achieved at epoch:',best_val_epoch,'\n')


def main():
    parser = argparse.ArgumentParser(description='Train patches')
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    argu_list = [[it for it in item.split(': ')] for item in args.list.split(', ')]
    dirpath = argu_list[0][1]
    learningrate = float(argu_list[1][1])
    batch_size = int(argu_list[2][1])
    max_epochs = int(argu_list[3][1])
    channels_out = int(argu_list[4][1])
    loss = argu_list[5][1]
    
    # get datasets
    patch_size = dirpath.split('/')[-2]
    train_odgt = dirpath + 'split/train/'+patch_size+'_train_patches.odgt'
    val_odgt = dirpath + 'split/val/'+patch_size+'_val_patches.odgt'

    
    train_ds = segmentation_dataset.SegmentationDataset(train_odgt,train=True)
    val_ds = segmentation_dataset.SegmentationDataset(val_odgt,train=False)
    
    
    # loader for training set 
    train_dl = DataLoader(dataset = train_ds,
                         batch_size = batch_size,
                         shuffle = True,
                         collate_fn = utils.collate_fn)
    # loader for validation set
    val_dl = DataLoader(dataset = val_ds,
                         batch_size = batch_size,
                         shuffle = True,
                         collate_fn = utils.collate_fn)
    
    # initialising the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(out_ch=channels_out)
    if torch.cuda.device_count()>1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    
    curtime = ('_').join(np.array(time.localtime()[:5],dtype=str))
    
    trainfile = open('./training_check_file_'+str(patch_size) + '_' + curtime,'w')
    trainfile.write('Patch size = '+str(patch_size))
    trainfile.write("\nUsing "+str(torch.cuda.device_count())+ " GPUs\n")
    trainfile.close()
    model.to(device)
    # print("\nModel summary: ",summary(model,input_size=(3, 512,512)))
    
    # define the loss function
    if loss == 'dice':
        criterion = loss_functions.DiceLoss().to(device)
    elif loss == 'generalised_dice':
        criterion = loss_functions.GeneralisedDiceLoss().to(device)
    else:
        print('not a type of loss')
        assert loss == 'dice'

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    
    run_training(train_dl, val_dl, model, criterion, optimizer,
                 batch_size, patch_size, curtime, device, max_epochs, save=True)
    
if __name__=="__main__":
    main()
