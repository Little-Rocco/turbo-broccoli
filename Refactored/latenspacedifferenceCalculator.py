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
from statistics import mean

path = 'LatentSpace'

#Remember to double check that the params are same for both the model e.g. dcgan and this!!!!
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int,   default=100,    help="dimensionality of the latent space")
parser.add_argument("--ngf",        type=int,   default=64,     help="Size of feature maps in generator")
parser.add_argument("--ndf",        type=int,   default=64,     help="Size of feature maps in discriminator")
parser.add_argument("--img_size",   type=int,   default=64,     help="size of each image dimension")
parser.add_argument("--channels",   type=int,   default=3,      help="number of image channels")
parser.add_argument("--modelNumber",   type=int,   default=100,      help="choice of model")

opt = parser.parse_args()


#Remember to double check that the generator is the same as used to train the model
class Generator(torch.nn.Module):
   def __init__(self):
      super(Generator, self).__init__()
      self.flatten = torch.nn.Flatten()
      self.linear_relu_stack = torch.nn.Sequential(
         # input is Z, going into a convolution
         torch.nn.ConvTranspose2d(opt.latent_dim, opt.ngf * 8, 4, 1, 0, bias=False),
         torch.nn.BatchNorm2d(opt.ngf * 8),
         torch.nn.ReLU(True),
         # state size. (ngf*8) x 4 x 4
         torch.nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
         torch.nn.BatchNorm2d(opt.ngf * 4),
         torch.nn.ReLU(True),
         # state size. (ngf*4) x 8 x 8
         torch.nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
         torch.nn.BatchNorm2d(opt.ngf * 2),
         torch.nn.ReLU(True),
         # state size. (ngf*2) x 16 x 16
         torch.nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
         torch.nn.BatchNorm2d(opt.ngf),
         torch.nn.ReLU(True),
         # state size. (ngf) x 32 x 32
         torch.nn.ConvTranspose2d(opt.ngf, opt.channels, 4, 2, 1, bias=False),
         torch.nn.Tanh()
         # state size. (nc) x 64 x 64
        )
   def forward(self, x):
      logits = self.linear_relu_stack(x).reshape((x.shape[0], opt.channels, opt.img_size, opt.img_size))
      return logits
      
def load_checkpoint(generator):
	model = torch.load('Models' + os.path.sep + 'Model' + str(opt.modelNumber) + '.pth')

	generator.load_state_dict(model['Generator'])

	fixed_noise = model['FixedNoise']
	
	generator.eval()
	device = None
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	generator.to(device)

def LSaverage(path):
   latentVectorList = []
   latentSpaceAverage = torch.zeros((1, opt.latent_dim, 1, 1))
   latentFileList = list(os.walk(path))
   for fName in latentFileList[0][2]:
      if(fName[-1] == 'h'):
         latentVector = torch.load(path +  os.path.sep + fName)['latentSpace']
         latentVector = [latentVector]
         latentVector = torch.stack(latentVector)
         latentVectorList.append(latentVector)

   latentVectorTensor = torch.stack(latentVectorList)
   return torch.mean(latentVectorTensor, 0)

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


generator = Generator()

load_checkpoint(generator)
colourPath = path + os.path.sep + "colors"

zRed = LSaverage(colourPath + os.path.sep + "Red" + os.path.sep)
zGreen = LSaverage(colourPath + os.path.sep + "Green" + os.path.sep)
zBlue = LSaverage(colourPath + os.path.sep + "Blue" + os.path.sep)

zBlack = LSaverage(colourPath + os.path.sep + "Black" + os.path.sep)
zWhite = LSaverage(colourPath + os.path.sep + "White" + os.path.sep)

zAvg = LSaverage(path + os.path.sep + "All" + os.path.sep)
zInput = LSaverage(path + os.path.sep + "Input" + os.path.sep)

############# Colour choosing #############
startRGB = [1.0, 1.0, 1.0]
targetRGB = [0.3, 0.0, 0.9]


############# latent vector calculation #############
Kdiff = mean([startRGB[0]-targetRGB[0], startRGB[1]-targetRGB[1], startRGB[2]-targetRGB[2]])
KWdiff = [Kdiff, -Kdiff]

newStartRGB = [startRGB[0]-Kdiff, startRGB[1]-Kdiff, startRGB[2]-Kdiff]
RGBdiff = [targetRGB[0]-newStartRGB[0], targetRGB[1]-newStartRGB[1], targetRGB[2]-newStartRGB[2]]

factor = 1.8
KWdiff = [KWdiff[0]*factor, KWdiff[1]*factor]
RGBdiff = [RGBdiff[0]*factor, RGBdiff[1]*factor, RGBdiff[2]*factor]

############# Image generation and showing #############
#fakeAvgImage = generator(zBlue)
#saveImage(fakeAvgImage)
fakeImageWithFeature = generator(zInput
                                 + RGBdiff[0]*zRed + RGBdiff[1]*zGreen + RGBdiff[2]*zBlue
                                 + KWdiff[0]*zBlack + KWdiff[1]*zWhite
                                 - (RGBdiff[0]+RGBdiff[1]+RGBdiff[2] + KWdiff[0]+KWdiff[1])*zAvg)

saveImage(fakeImageWithFeature)