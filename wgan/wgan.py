# Documentation
# install for IPython: https://ipython.org/install.html 
# "hack" to allow for loading sibling packages: https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944
# 
# end of documentation


#the hack for loading sibling packages
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"
    
from Utility.graphHelper import *
from Utility.dataHelper import *

#reset path and package
path.pop()
__package__ = None



import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

#from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime

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
parser.add_argument("--sample_interval", type=int,  default=100,    help="interval betwen image samples")
#add argument: update_interval
#add argument: epochs_per_save

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

dataroot = "C:\\Users\\Anders\\source\\repos\\data\\Fruits_360\\Training"

dataset = dataLoad.loadImages(dataroot, opt.img_size)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
iters_per_epoch = (math.ceil(len(dataloader.dataset.imgs)/opt.batch_size))

cuda = True if torch.cuda.is_available() else False


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
        #.reshape((z.shape[0], opt.channels, opt.img_size, opt.img_size))
        #img = img.view(img.shape[0], *img_shape)
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
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# # Configure data loader
# #os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
#Lists for plotting
gen_imgs = []
generator_losses = []
generator_losses_x = []
discriminator_losses = []
discriminator_losses_x = []
real_losses = []
real_losses_x = []

batches_done = 0
accuracy_real, accuracy_fake = 0, 0
batches_done = 0

for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        #Pass real imgs through discriminator
        real_preds = -torch.mean(discriminator(real_imgs))
        #real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_loss = -torch.mean(real_preds)
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim, 1, 1))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs.append(generator(z))
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs[-1]))

            loss_G.backward()
            optimizer_G.step()

            #print(
            #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            #)
            
            #real_loss (D(x)) = discriminator output on real data
            #loss_D.item (critic loss) = D(x) - D(G(z))
            #loss_G.item (generator loss) = D(G(z)
            if i % 10 == 0:
                print('[%5d/%5d][%5d/%5d]\tD(x): %.4f\tCritic loss: %.4f\tGenerator loss: %.4f\t'
                    % (epoch, opt.n_epochs,
                        i, len(dataloader),
                        real_loss,
                        loss_D.item(),
                        loss_G.item(),))
                # For time stamps
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
        
        #For plotting
        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())
        real_losses.append(real_loss.item())

        generator_losses_x.append(batches_done / iters_per_epoch)
        discriminator_losses_x.append(batches_done / iters_per_epoch)
        real_losses_x.append(batches_done / iters_per_epoch)

        

        if batches_done % opt.sample_interval == 0:
            real_batch = next(iter(dataloader))

            #save graph for loss
            g_loss_curve = curve(generator_losses_x, generator_losses, "Generator Loss (D(G(z)))")
            d_loss_curve1 = curve(discriminator_losses_x, discriminator_losses, "Critic Loss, Fake (D(x) - D(G(z)))")
            d_loss_curve2 = curve(real_losses_x, real_losses, "Critic Loss, Real (D(x))")
            loss_graph = graph([g_loss_curve, d_loss_curve1, d_loss_curve2], "Epochs", "Loss", "Generator and Discriminator Loss During Training")
            saver.saveGraph(loss_graph,
                            directory="images",
                            filename="plot_%d.png" % batches_done)


            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # save the images
            saver.saveImages(real_batch, gen_imgs,
                             directory="images", 
                             filename="%d.png" % batches_done,
                             device="cpu")

        batches_done += 1
