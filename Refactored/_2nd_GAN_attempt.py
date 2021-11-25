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


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",   type=int,   default=1000,   help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=64,     help="size of the batches")
parser.add_argument("--lr",         type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu",      type=int,   default=8,      help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=100,    help="dimensionality of the latent space")
parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
parser.add_argument("--channels",   type=int,   default=1,      help="number of image channels")
parser.add_argument("--n_critic",   type=int,   default=1,      help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=-1,     help="lower and upper clip value for disc. weights. (-1 = no clipping)")
parser.add_argument("--sample_interval", type=int,  default=200,    help="iters between image samples")
parser.add_argument("--update_interval", type=int,  default=100,     help="iters between terminal updates")
parser.add_argument("--epochs_per_save", type=int,  default=10,      help="epochs between model saves")
parser.add_argument("--split_disc_loss", type=bool,  default=False,  help="whether to split discriminator loss into real/fake")
parser.add_argument("--beta1",      type=float, default=0.5,    help="beta1 hyperparameter for Adam optimizer")

opt = parser.parse_args()
print(opt)


dataroot = "C:\\Users\\Anders\\source\\repos\\data\\shapes"
seed = torch.Generator().seed()
print("Current seed: " + str(seed))

engine = Engine(opt, dataroot, seed, 0.7)



class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            # the neural network
            torch.nn.Linear(opt.latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, opt.img_size * opt.img_size * opt.channels),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x).reshape((x.shape[0], opt.channels, opt.img_size, opt.img_size))
        return logits


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(opt.img_size * opt.img_size * opt.channels, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Initialize generator and discriminator
engine.add_networks(Generator(), Discriminator())

# Loss function
loss_function = torch.nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizer_discriminator = torch.optim.Adam(engine.discriminator.parameters(), lr=engine.opt.lr,
                                           betas=(engine.opt.beta1, 0.999))
optimizer_generator = torch.optim.Adam(engine.generator.parameters(), lr=engine.opt.lr,
                                       betas=(engine.opt.beta1, 0.999))



# Add functions and run
engine.add_functions(optimizer_generator, optimizer_discriminator, loss_function)
engine.run()