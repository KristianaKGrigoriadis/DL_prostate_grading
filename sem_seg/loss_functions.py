import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, model_output, target):
        num_classes = model_output.shape[1]
        score = model_output.clone()
        probas = score.clone()
        target = target.type(torch.LongTensor)
        target_1_hot = torch.eye(num_classes)[target.squeeze(1)]

        target_1_hot = target_1_hot.type(probas.type())        
        target_1_hot = target_1_hot.reshape(probas.shape)

        dims = (0,) + tuple(range(2, probas.ndimension()))
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
        probas = score.clone()

        target = target.type(torch.LongTensor)
        target_1_hot = torch.eye(num_classes)[target.squeeze(1)]

        target_1_hot = target_1_hot.type(score.type())
        target_1_hot = target_1_hot.reshape(probas.shape)

        dims = tuple(range(2, probas.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)
        weights = 1./((probas**2).sum(dim=dims))
        
        dice_loss = (2. * torch.sum(weights * intersection))/(torch.sum(weights*cardinality) + self.eps)
        return (1 - dice_loss)

def intersectionAndUnion(imPred, imLab, numClass, ignore_index=-1):
    """ Adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/eval.py"""
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()
    # ignore rare/non existing especially for cityscape
    ignore_label = imLab != ignore_index
    imLab = imLab[ignore_label]
    imPred = imPred[ignore_label]
    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    imPred = imPred * (imLab > 0)
    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
  
    return (area_intersection, area_union, area_lab)

