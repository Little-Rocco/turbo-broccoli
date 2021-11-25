# Documentation
# install for IPython: https://ipython.org/install.html 
#
#
# end of documentation



from Utility.graphHelper import *
from Utility.dataHelper import *
from Utility.Engine import *



import argparse
import os
import torch

import torch.nn as nn


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",   type=int,   default=1000,   help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=64,     help="size of the batches")
parser.add_argument("--lr",         type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu",      type=int,   default=8,      help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=64,     help="dimensionality of the latent space")
parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
parser.add_argument("--channels",   type=int,   default=3,      help="number of image channels")
parser.add_argument("--n_critic",   type=int,   default=1,      help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,   help="lower and upper clip value for disc. weights. (-1 = no clipping)")
parser.add_argument("--sample_interval", type=int,  default=100,    help="iters between image samples")
parser.add_argument("--update_interval", type=int,  default=10,    help="iters between terminal updates")
parser.add_argument("--epochs_per_save", type=int,  default=5,    help="epochs between model saves")

opt = parser.parse_args()
print(opt)


dataroot = "C:\\Users\\Anders\\source\\repos\\data\\Fruits_360\\Training"
seed = torch.Generator().seed()
print("Current seed: " + str(seed))

engine = Engine(opt, dataroot, seed, 0.7)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( opt.latent_dim, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Linear(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Initialize generator and discriminator
engine.add_networks(Generator(), Discriminator())

# Loss function
def meanLoss(output, labels):
    return torch.mean(output)*int((0.5-labels[-1])*2)

# Optimizers
optimizer_G = torch.optim.RMSprop(engine.generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(engine.discriminator.parameters(), lr=opt.lr)

# Add functions and run
engine.add_functions(optimizer_G, optimizer_D, meanLoss)
engine.run()