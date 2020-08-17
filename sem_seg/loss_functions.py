import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, model_output, target):
        # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        num_classes = model_output.shape[1]
        score = model_output.clone()
        #score = softmax(score)
        probas = score.clone()
        #print("torch.sum(torch.isnan(probas)) =",torch.sum(torch.isnan(probas)))
        target = target.type(torch.LongTensor)
        target_1_hot = torch.eye(num_classes)[target.squeeze(1)]

        target_1_hot = target_1_hot.type(probas.type())        
        target_1_hot = target_1_hot.reshape(probas.shape)

        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)

        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
    
    
class GeneralisedDiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(GeneralisedDiceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, model_output, target):
        num_classes = model_output.shape[1]
        score = model_output.clone()        
        #score = softmax(score)
        probas = score.clone()

        target = target.type(torch.LongTensor)
        target_1_hot = torch.eye(num_classes)[target.squeeze(1)]

        target_1_hot = target_1_hot.type(score.type())
        target_1_hot = target_1_hot.reshape(probas.shape)

        dims = tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)
        weights = 1./((probas**2).sum(dim=dims))
        
        dice_loss = (2. * torch.sum(weights * intersection))/(torch.sum(weights*cardinality) + self.eps)
        return (1 - dice_loss)



