import torch

def softmax(torch_tensor):
    numerator = torch.exp(torch_tensor)
    denominator = numerator.sum(dim=1).unsqueeze(1)
    output = numerator/denominator
    return output


def collate_fn(batch):
    images = []
    labels = []
    for i,(image, label) in enumerate(batch):
        images.append(image)
        labels.append(label)
        # print(i,label.unique())

    images_t = torch.cat(images).reshape((len(images),) + tuple(images[0].shape))
    labels_t = torch.cat(labels).reshape((len(labels),) + tuple(labels[0].shape))
    
    images_t = images_t.float()

    return images_t, labels_t



def accuracy(model_output, labels):
    probs = softmax(model_output)
    preds = probs.argmax(1)
    correct = (preds == labels).sum().item()
    total = (preds == preds).sum().item()
    acc = correct/total
    return acc
    
    
