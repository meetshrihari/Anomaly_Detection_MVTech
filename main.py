import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from glob import glob
import datetime
import cv2

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
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
#------------------ 
from features.model import ResNet_, Decod, VAE, Discriminator, AdaCos
from features.my_block import My_BasicBlock
from features.train import vae_gan_train, train, validate
from features.utils import MVTec_dataset, gen_dataset, plot_tSNE, plot_tSNE_MVtec




'''
Initilize the Model with Backbone network, dataset, batchsize etc
'''

#def parse_args():
parser = argparse.ArgumentParser()

parser.add_argument('--name', default='VAE_GAN', help='model name')
parser.add_argument('--arch', default='ResNet18')
parser.add_argument('--backbone', default='resnet18')
parser.add_argument('--dataset', default='leather') # capsule, leather ========== type of Dataset ==========
parser.add_argument('--metric', default='adacos')
parser.add_argument('--num-features', default=512, type=int,
                    help='dimention of embedded features')
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
parser.add_argument('--min-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--learning', default='unsupervised') # semi-supervised

args = parser.parse_args()

#  --name 'VAE_GAN' --arch 'ResNet18' --dataset 'leather' --batch-size 8 --epochs 250 --learning-rate 1e-1 --min-lr 1e-3 --momentum 0.9 --weight-decay 1e-4 --learning 'unsupervised'
# for passing arguments
import sys
if sys.argv:
        del sys.argv[1:]

#args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print('Details of System ------------------------')
for arg in vars(args):
    print('%s: %s' % (arg, getattr(args, arg)))
print('----------------------------------------')

#----------------- Dataset -----------
train_img_paths,train_labels, test_img_paths, test_labels = gen_dataset(args)

transform_mvtec_train = transforms.Compose([transforms.Resize((300,300)),
                                            transforms.RandomCrop((256,256)),
                                            #transforms.RandomHorizontalFlip(p=0.5),
                                            #transforms.RandomVerticalFlip(p=0.5),
                                            transforms.ToTensor()])

transform_mvtec_test = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_set = MVTec_dataset(train_img_paths, train_labels, transform=transform_mvtec_train)

test_set = MVTec_dataset(test_img_paths, test_labels, transform=transform_mvtec_test)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0)
#-------------------------------------

D = Discriminator()
model = VAE(args)
resN = ResNet_(args)
model = model.to(device)
D = D.to(device)
resN = resN.to(device)
metric_fc = AdaCos(num_features=args.num_features, num_classes=args.num_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)

cudnn.benchmark = True


optimizer = optim.SGD(filter(lambda p: p.requires_grad, resN.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                            T_max=args.epochs, eta_min=args.min_lr)

log = pd.DataFrame(index=[], columns=[
      'epoch', 'lr', 'loss', 'acc@1', 'acc@5', 'val_loss', 'val_acc1', 'val_acc5'
    ])




best_loss = float('inf')


for epoch in range(args.epochs):
    print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

    scheduler.step()

    # train for one epoch
    train_log, Error_log = train(args, train_loader, resN, model,D, metric_fc, criterion, optimizer, epoch)
    # evaluate on validation set
    val_log = validate(args, test_loader,resN, model, metric_fc, criterion, epoch)

    print('loss %.4f - acc@1 %.4f - acc@5 %.4f - val_loss %.4f - val_acc@1 %.4f - val_acc@5 %.4f'
            %(train_log['loss'], train_log['acc@1'], train_log['acc@5'], val_log['loss'], val_log['acc@1'], val_log['acc@5']))

    tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc@1'],
            train_log['acc@5'],
            val_log['loss'],
            val_log['acc@1'],
            val_log['acc@5'],
        ], index=['epoch', 'lr', 'loss', 'acc@1', 'acc@5', 'val_loss', 'val_acc1', 'val_acc5'])

    log = log.append(tmp, ignore_index=True)

    if val_log['loss'] < best_loss:
        torch.save(model.state_dict(), 'vae_model_best.pt')
        torch.save(D.state_dict(), 'D_model_best.pt')
        torch.save(metric_fc.state_dict(), 'metric_best.pt')
        best_loss = val_log['loss']
        print("=> saved best model")

plt.title("Encoder, Decoder (Generator) and Discriminator Loss During Training")
plt.plot(Error_log['G_Losses'], label = "G")
plt.plot(Error_log['D_Losses'], label = "D")
plt.plot(Error_log['E_Losses'], label = "E")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.load_state_dict(torch.load('vae_model_best.pt'))
metric_fc.load_state_dict(torch.load('metric_best.pt'))

#...................

X_train, y_train, plt_arc_df_train = plot_tSNE_MVtec(resN, metric_fc, train_loader)
X_test, y_test, plt_arc_df_test = plot_tSNE_MVtec(resN, metric_fc, test_loader)
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])
plot_tSNE(X, y)

#.......................

list(model.resnet.children())[1][6][1] = My_BasicBlock(512, 512)

params = torch.load('vae_model_best.pt')
model.load_state_dict(params)

#------------------
transform_test_bad = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_test_set_bad = MVTec_dataset(
        test_img_paths,
        test_labels,
        transform=transform_test_bad)

test_test_set_bad = MVTec_dataset(
        test_img_paths,
        test_labels,
        transform=transform_test_bad)

train_test_bad_loader = DataLoader(train_test_set_bad, batch_size=len(train_test_set_bad))
train_test_bad_dataset_array = next(iter(train_test_bad_loader))[0].numpy()
train_test_bad_dataset_tensor = torch.from_numpy(train_test_bad_dataset_array)

test_test_bad_loader = DataLoader( test_test_set_bad, batch_size=len( test_test_set_bad))
test_test_bad_dataset_array = next(iter( test_test_bad_loader))[0].numpy()
test_test_bad_dataset_tensor = torch.from_numpy( test_test_bad_dataset_array)


def Grad_Cam_Leather(train_test_bad_dataset_tensor, train_img_paths_test_bad):
    for i in range(len(train_test_bad_dataset_tensor)):
        print(train_img_paths_test_bad[i])
        img = train_test_bad_dataset_tensor[i]
        original_img = img  # To display image at last
        img = img.unsqueeze(0)
        model.eval()
        metric_fc.eval()

        temp_output = model.encoder(img)
        output = metric_fc(temp_output)
        # print(output)
        class_label = output.argmax(dim=1).item()
        class_score = output[0][1]
        med_out = My_BasicBlock.get_med_out()
        N, C, H, W = med_out.shape
        grads = torch.autograd.grad(class_score, med_out)
        w = grads[0][0].mean(-1).mean(-1)  # Calculate the weight by averaging the gradient for each channel
        ans = torch.matmul(w, med_out.view(C, H * W))
        ans = F.relu(ans)
        ans = ans.view(H, W).cpu().detach().numpy()
        plt.imshow(ans)
        ans = cv2.resize(ans, (256, 256))
        plt.imshow(original_img.permute(1, 2, 0))
        plt.show()
        plt.imshow(original_img.permute(1, 2, 0))
        plt.imshow(ans, alpha=0.5, cmap='jet')
        plt.show()
        #if i == 5:
        #    break

print('GRAD CAM PRINTING')
model.cpu()
Grad_Cam_Leather(train_test_bad_dataset_tensor, test_img_paths)