from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import math
from PIL import Image
from glob import glob
import datetime
#import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
#from torch.utils.data import
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms, models
from collections import OrderedDict
from sklearn.manifold import TSNE
from torchvision.utils import make_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_Losses = []
G_Losses = []
E_Losses = []


def loss_function(recon_x, x, mean, logstd):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # Because var is the natural logarithm of the standard deviation,
    # first find the natural logarithm and then square it to convert to variance
    var = torch.pow(torch.exp(logstd), 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return BCE + KLD


def vae_gan_train(args, input, vae, D, optimizer):
    ###################################################################
    #           Initilization
    ###################################################################
    z_dim = 512
    nz = z_dim

    # print("Random Seed: 11")
    random.seed(11)
    torch.manual_seed(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = input.to(device)
    # Can optimize operating efficiency
    cudnn.benchmark = True

    vae = vae.to(device)
    D = D.to(device)

    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss()

    # print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0001, weight_decay=5e-4)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001, weight_decay=5e-4)

    gen_win = None
    rec_win = None

    ###################################################
    # (2) Update G network which is the decoder of VAE
    ###################################################
    recon_data, mean, logstd = vae(data)
    # print('recon Data =  ',recon_data.shape,'original Data =  ', data.shape)
    vae.zero_grad()
    vae_loss = loss_function(recon_data, data, mean, logstd)
    vae_loss.backward(retain_graph=True)
    optimizerVAE.step()

    ###################################################################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###################################################################
    # train with real
    D.zero_grad()

    # label = label.to(device)
    batch_size = data.shape[0]
    output = D(data)
    # print("output size of netDs: ", output.size())

    real_label = (torch.ones(batch_size) * 0.95).to(device)  # Define the real image label as 1
    fake_label = (torch.ones(batch_size) * 0.05).to(device)  # Define the label of the fake picture as 0
    errD_real = criterion(output, real_label)
    errD_real.backward()
    real_data_score = output.mean().item()

    # train with fake, taking the noise vector z as the input of D network
    # Randomly generate a latent variable, and then generate a picture through the decoder
    z = torch.randn(batch_size, nz).to(device)
    # Turn the latent variable z into a fake picture through vae's decoder
    fake_data = vae.decode(z)
    # print(fake_data.shape)
    output = D(fake_data)
    errD_fake = criterion(output, fake_label)
    errD_fake.backward()
    # fake_data_score Used to output the score of the fake photo, 0 is the most false, 1 is true
    fake_data_score = output.data.mean()
    errD = errD_real + errD_fake  # Discriminator Loss
    optimizerD.step()

    ###############################################
    # (3) Update G network: maximize log(D(G(z)))
    ###############################################
    vae.zero_grad()
    real_label = torch.ones(batch_size).to(device)  # Define the real image label as 1
    recon_data, mean, logstd = vae(data)
    output = D(recon_data)
    errVAE = criterion(output, real_label)
    abs_loss = l1_loss(recon_data, data)  # Generator Loss
    errVAE.backward()
    D_G_z2 = output.mean().item()
    optimizerD.step()

    ############################################### End of Training Code

    D_Losses.append(errD.item())
    G_Losses.append(abs_loss.item())

    z = vae.encoder(data)
    return z, real_data_score, fake_data_score, errD.item(), errVAE.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args, train_loader,resN, model,D, metric_fc, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    model.train()
    resN.train()
    metric_fc.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.long().to(device) 

        #********************** VAE Training *******************
        z, real_score, fake_score, errD, errVAE = vae_gan_train(args, input, model, D, optimizer )
        #*********************************************

        for i in range(3):
            feature = resN(input)
            output = metric_fc(feature, target)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))
        acc5s.update(acc5.item(), input.size(0))
        E_Losses.append(loss.item())

        # compute gradient and do optimizing step based on Adacos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc@1', acc1s.avg),
        ('acc@5', acc5s.avg),
    ])
    Error_log = OrderedDict([
        ('D_Losses', D_Losses),
        ('G_Losses', G_Losses),
        ('E_Losses', E_Losses),
    ])

    print('real_score: %.4f fake_score: %.4f Loss_D: %.4f errVAE: %4f'
          % (real_score, fake_score, errD, errVAE))

    return log, Error_log

def validate(args, val_loader,resN, model, metric_fc, criterion, epoch):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    # switch to evaluate mode
    model.eval()
    resN.eval()
    metric_fc.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.long().to(device) 

            feature = resN(input)
            output = metric_fc(feature)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))
            acc5s.update(acc5.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc@1', acc1s.avg),
        ('acc@5', acc5s.avg),
    ])

    #if epoch %5 == 0:
    x_0,_ = iter(val_loader).next()
    recon_x,_,_ = model(x_0.to(device))
    fake_images = make_grid(recon_x.cpu(), nrow=8, normalize=True).detach()
    plt.imshow(np.transpose((fake_images), [1, 2, 0])[:,:,0:3], interpolation="bicubic")
    plt.show()



    return log