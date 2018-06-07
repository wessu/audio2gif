import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from miscc.datasets import AudioSetImage
import os

TRAIN = True
SAVE_MODEL = True
VAL = False
MODEL_NAME = '../data/inception_v3_pretrain'
MAX_EPOCH = 10
data_dir = '../data/audioset/train/feature/melspec/wrap_all'

img_dir = '../data/audioset/val/real'


def inception_score(imgs, inception_model=None, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    modified from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    if inception_model == None:
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    dataset = AudioSetImage(data_dir)
    
    cuda = True
    if cuda:
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(1)
    else:
        dtype = torch.FloatTensor
    if TRAIN:
        model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
        for epoch in range(MAX_EPOCH):
            for i, data in enumerate(dataloader):
                img, label = data
                img = img.type(torch.FloatTensor)                
                label = img.type(torch.FloatTensor)
                if cuda:
                    img = img.cuda()
                    label = label.cuda()
                img = up(img)
                optimizer.zero_grad()
                pred = model(img)
                print(pred)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
            predicted_label = torch.argmax(pred, dim=1)
            acc = (predicted_label == label)/len(label)
            print("Epoch {}/{}, loss {}, acc {} ".format(epoch, MAX_EPOCHi, loss, acc))

        if SAVE_MODEL:
            torch.save(model, MODEL_NAME)
    if VAL:
        with open(MODEL_NAME, 'r') as f:
            model = torch.load(f)
        print("Evaluating dataset from {}".format(data_dir))
        dataset = AudioSetImage(data_dir)
        score, std = inception_score(dataset, model)
        print("Inception score: {}, std {}".format(score, std))








