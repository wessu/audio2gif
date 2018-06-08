import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision.models.vgg import vgg16_bn
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from miscc.datasets import AudioSetImage, iterate_minibatches, ImageSet
import os

TRAIN = True
SAVE_MODEL = True
VAL = False
MODEL_NAME = '../data/vgg_pretrain'
MAX_EPOCH = 100
data_dir = '../data/audioset/train/feature/melspec/wrap_all'
eval_dir = '../data/audioset/eval/feature/melspec/wrap_all'
#data_dir = ["/ext3","/ext2"] 
img_dir = '../data/audioset/val/real'


def inception_score(fp):
    inception_model = torch.load(MODEL_NAME)
    real_set = ImageSet(fp, False)
    fake_set = ImageSet(fp, True)
    r_loss, r_acc = evaluation(real_set, inception_model)
    f_loss, f_acc = evaluation(fake_set, inception_model)

    print('Real acc {}, loss {}'.format(r_acc, r_loss))
    print('Fake acc {}, loss {}'.format(f_acc, f_loss))

    
def evaluation(eval_set, model):
    eval_dataloader = torch.utils.data.DataLoader(eval_set, batch_size=10)
    num_correct = 0
    num_samples = 0
    n_iters = 0
    accu_loss = 0
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    up = nn.Upsample(size=(224, 224), mode='bilinear').type(torch.cuda.FloatTensor)
    for i, data in enumerate(eval_dataloader):
        img, label = data
        img = torch.tensor(img).type(torch.FloatTensor)                
        label = torch.tensor(label).type(torch.LongTensor)
        img = img.cuda()
        label = label.cuda()
        img = up(img)
        pred = model(img)
        loss = loss_fn(pred, label)
        n_iters += 1
        with torch.no_grad():
            predicted_label = torch.argmax(pred, dim=1)
            num_correct += (predicted_label == label).sum()
            num_samples += label.size(0)
            accu_loss += loss.item()
    acc = float(num_correct) * 100.0 / float(num_samples)
    loss = accu_loss/n_iters
    print(num_correct, num_samples)
    print("Eval: loss {:.4f}, acc {:.2f}% ".format(loss, acc))
    return loss, acc

if __name__ == '__main__':
    print("Loading audioset")
    train_set = AudioSetImage(data_dir)
    eval_set = AudioSetImage(eval_dir)
    #dataset2 = AudioSetImage(data_dir[1])
    audio_label_dim = 10
    # print("starting")
    cuda = True
    if cuda:
        dtype = torch.cuda.FloatTensor
        print('cuda device:', torch.cuda.current_device())
    else:
        dtype = torch.FloatTensor
    if TRAIN:
        print("loading pre trained")
        full_inception = vgg16_bn(pretrained=True).type(dtype)
        removed = list(full_inception.classifier.children())[:-1]
        # print(removed[-1].size())
        full_inception.classifier = nn.Sequential(*removed, nn.Linear(4096, audio_label_dim))
        model = full_inception
        model.train()
        if cuda:
            model.cuda()
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=10)
        #dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=64)
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        up = nn.Upsample(size=(224, 224), mode='bilinear').type(dtype)

        print('Start training')
        lowest_eval_loss = float('inf')
        for epoch in range(MAX_EPOCH):
            print('')
            num_correct = 0
            num_samples = 0
            for i, data in enumerate(train_dataloader):
                img, label = data
                img = torch.tensor(img).type(torch.FloatTensor)                
                label = torch.tensor(label).type(torch.LongTensor)
                if cuda:
                    img = img.cuda()
                    label = label.cuda()
                img = up(img)
                optimizer.zero_grad()
                pred = model(img)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                
                # Statistics
                if i % 50 == 0:
                    print('Epoch %d, Iteration %d, loss = %.4f' % (epoch, i, loss.item()))
                with torch.no_grad():
                    predicted_label = torch.argmax(pred, dim=1)
                    num_correct += (predicted_label == label).sum()
                    num_samples += label.size(0)
            acc = float(num_correct) * 100.0 / float(num_samples)
            print("Epoch {}/{}, acc {:.2f}% ".format(epoch, MAX_EPOCH, acc))

            # Evaluation
            eval_loss, eval_acc = evaluation(eval_set, model)
            if eval_loss < lowest_eval_loss:
                torch.save(model, MODEL_NAME)


