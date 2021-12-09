from numpy.lib.function_base import append
import torch
import os
import argparse
from torch.autograd import variable
from torch.cuda import is_available
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision

path = 'Refactored' + os.path.sep + 'LatentSpace' + os.path.sep

#Remember to double check that the params are same for both the model e.g. dcgan and this!!!!
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int,   default=100,    help="dimensionality of the latent space")
parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
parser.add_argument("--channels",   type=int,   default=3,      help="number of image channels")
parser.add_argument("--modelNumber",   type=int,   default=0,      help="choice of model")

opt = parser.parse_args()


#Remember to double check that the generator is the same as used to train the model
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


def load_checkpoint(generator):
	model = torch.load('Models' + os.path.sep + 'Model' + opt.modelNumber + '.pth')

	generator.load_state_dict(model['Generator'])

	fixed_noise = model['FixedNoise']
	
	generator.eval()
	device = None
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	generator.to(device)

def LSaverage():
   latentSpaceAverage = torch.zeros((64, opt.latent_dim, 1, 1))
   i = 0

   for fName in list(os.walk(path))[0][2]:
      latentSpace = torch.load(path +  os.path.sep + fName)['latentSpace']
      latentSpaceAverage += latentSpace
      i += 1

   return latentSpaceAverage/i

def saveImage(img):
   channels = opt.channels
   device = None
   if torch.cuda.is_available():
      device = 'cuda:0'
   else:
      device = 'cpu'

   plt.figure(figsize=(1, 1), dpi=83*5)
   img_0 = img.to(device)
   if(channels == 1):
      plt.imshow(
            np.transpose(torchvision.utils.make_grid(img_0[0], padding=0, normalize=True).cpu(),
                  (0, 1)))
   else:
      plt.imshow(
            np.transpose(torchvision.utils.make_grid(img_0, padding=0, normalize=True).cpu(),
                  (1, 2, 0)))
   plt.show()


z = LSaverage()
generator = Generator()
fakeImage = generator(z)
saveImage(fakeImage)

