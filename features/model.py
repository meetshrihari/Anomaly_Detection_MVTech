import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torchvision import datasets, transforms, models
import math
from torch.nn import Parameter


class ResNet_(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        last_channels = 512
        # last_channels = 2

        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4)

        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(16 * 16 * last_channels, args.num_features)
        # self.fc = nn.Linear(524288, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)

        return output


class Decod(nn.Module):
    def __init__(self):
        super(Decod, self).__init__()

        self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.layer2_6 = nn.Sequential(
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True), # to add the layer in the place
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.layer8_12 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.layer13 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.layer14_18 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.layer19 = nn.ConvTranspose2d(in_channels=128 , out_channels=64, kernel_size=4, stride=2, padding=1)
        self.layer20_24 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer25_26 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 , out_channels=3, kernel_size=3, stride=1, padding=1),
            # this is for gray images
            #nn.ConvTranspose2d(in_channels=64 , out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print('start of Decoder ----------------------------------------------------')
        x1 = self.layer1(input)
        x2_6 = self.layer2_6(x1)
        x1_plus_6 = x1 + x2_6
        #print("1 = ", x1_plus_6.shape)

        x7 = self.layer7(x1_plus_6)
        x8_12 = self.layer8_12(x7)
        x7_plus_12 = x7 + x8_12
        #print("2 = ", x7_plus_12.size())

        x13 = self.layer13(x7_plus_12)
        x14_18 = self.layer14_18(x13)
        x13_plus_18 = x13 + x14_18
        #print("3 = ", x13_plus_18.size())

        x19 = self.layer19(x13_plus_18)
        x20_24 = self.layer20_24(x19)
        x19_plus_24 = x19 + x20_24
        #print("4 = ", x19_plus_24.size())

        x25_26 = self.layer25_26(x19_plus_24)
        #print('End of Decoder ----------------------------------------------------')

        return x25_26



class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        # encoder ********************************
        self.resnet = ResNet_(args)

        '''self.E_layer1_8 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1),
                                      nn.BatchNorm2d(num_features=128),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1),
                                      nn.BatchNorm2d(num_features=256),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1),
                                      nn.BatchNorm2d(num_features=512),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1))
        self.E_bn_relu = nn.Sequential(nn.BatchNorm2d(num_features=512),
                                      nn.ReLU(inplace=True))
                                      '''

        # *******************************************************

        # decoder
        self.dim = 512 * 16 * 16
        self.fc_d = nn.Sequential(
            nn.Linear(args.num_features, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = Decod()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 512, 16, 16)
        return self.decoder(h)

    def encoder(self, x):
        x = self.resnet(x)  # ouput each = (8,512)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x), self.encoder(x)  # ouput each = (8,512)
        # print('sizes ---------- ', mu.shape, logvar.shape)
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


'''
# custom weights initialization called on G and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    if classname.find('ConvTranspose') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
'''

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()

      self.dim = 512*8*8
      self.layer1_13 = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
          # for gray images
          #nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1),
          nn.BatchNorm2d(num_features=128),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1),
          nn.BatchNorm2d(num_features=256),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1),
          nn.BatchNorm2d(num_features=512),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1),
          nn.Sigmoid()
      )
      self.linear = nn.Sequential(nn.Linear(self.dim, 1024),
                                  nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Linear(1024, 1),
                                  nn.Sigmoid())

    def forward(self, input):
      x = self.layer1_13(input)
      x = x.view(x.size(0), -1)
      #x=x.view(-1, 512*8*8)
      x= self.linear(x)
      return x.squeeze(1)


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W) # Applies a linear transformation to the incoming data: y = xA^T + b
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)) # bring all the point from -1 to 1
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output