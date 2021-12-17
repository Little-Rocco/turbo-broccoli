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
from math import sqrt, pi

path = 'LatentSpace'
startRGB = [1.0, 0.9, 1.0]
targetRGB = [1.0, 0.4, 0.0]
factor = 1.0


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

zBlack = LSaverage(colourPath + os.path.sep + "Black" + os.path.sep)
zKrgb = [0.15, 0.15, 0.15]
zWhite = LSaverage(colourPath + os.path.sep + "White" + os.path.sep)
zWrgb = [0.8, 0.8, 0.8]

zOrange = LSaverage(colourPath + os.path.sep + "Orange" + os.path.sep)
zOrgb = [0.6, 0.4, 0.2]
zPurple = LSaverage(colourPath + os.path.sep + "Purple" + os.path.sep)
zMrgb = [0.4, 0.25, 0.7]
zYellow = LSaverage(colourPath + os.path.sep + "Yellow" + os.path.sep)
zYrgb = [0.8, 0.8, 0.5]

zRed = LSaverage(colourPath + os.path.sep + "Red" + os.path.sep)
zRrgb = [0.7, 0.4, 0.4]
zGreen = LSaverage(colourPath + os.path.sep + "Green" + os.path.sep)
zGrgb = [0.25, 0.75, 0.6]
zBlue = LSaverage(colourPath + os.path.sep + "Blue" + os.path.sep)
zBrgb = [0.15, 0.25, 0.6]

zAvg = LSaverage(path + os.path.sep + "All" + os.path.sep)
zInput = LSaverage(path + os.path.sep + "Input" + os.path.sep)


############# latent vector calculation #############
def vecLength(vec):
    return sqrt((vec[0]**2) + (vec[1]**2) + (vec[2]**2))

def vecAngle(vec1, vec2):
    # Returns the angle between the vectors, in radians in range [0, pi]
    lenThing = vecLength(vec1) * vecLength(vec2)
    if lenThing != 0:
        return np.arccos((vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]) / (vecLength(vec1) * vecLength(vec2)))
    else:
        return 0

def findClosestVec(desiredVec, vecList):
    angles = []
    for vec in vecList:
        angles.append(vecAngle(desiredVec, vec))
    
    lowestAngle = 4
    angleIdx = 0
    i = 0
    for a in angles:
        if a < lowestAngle or pi-a < lowestAngle:
            lowestAngle = min([a, pi-a])
            angleIdx = i
        i += 1
    return angleIdx


def findWeights(desiredVec, vecList):
    weights = []
    for vec in vecList:
        weights.append(0)

    # use each vector once
    remainingVecList = vecList
    currentVec = [0, 0, 0]
    remainingVec = desiredVec
    for i in range(len(vecList)):
        # find closest remaining vector
        idx = findClosestVec(remainingVec, remainingVecList)
        vec = remainingVecList[idx]

        # find whether to use positive or negative amount
        sign = 1
        angle = vecAngle(currentVec, vec)
        if angle > pi/2:
            sign = -1
        vec = [vec[0]*sign, vec[1]*sign, vec[2]*sign]

        # find ideal amplitude
        vecLen = vecLength(vec)
        normVec = [vec[0]/vecLen, vec[1]/vecLen, vec[2]/vecLen]
        amplitude = np.dot(remainingVec, normVec)

        # add new vector
        mult = sign*amplitude
        currentVec = [currentVec[0]+(vec[0]*mult), currentVec[1]+(vec[1]*mult), currentVec[2]+(vec[2]*mult)]
        remainingVec = [desiredVec[0]-currentVec[0], desiredVec[1]-currentVec[1], desiredVec[2]-currentVec[2]]

        # save weight
        weights[idx] = mult

        # remove element to only use it once
        remainingVecList.pop(idx)

    return weights


def getLatent(initialRGB, targetRGB, RGBvecList, initialLatent, latentVecList, latentAvg):
    desiredVector = [targetRGB[0]-initialRGB[0], targetRGB[1]-initialRGB[1], targetRGB[2]-initialRGB[2]]
    weights = findWeights(desiredVector, RGBvecList)

    newLatent = initialLatent
    totalWeight = 0
    for i in range(len(weights)):
        newLatent += weights[i]*latentVecList[i]*factor
        totalWeight += weights[i]*factor

    return newLatent-(totalWeight*latentAvg)


RGBvecList = [zKrgb, zWrgb, zOrgb, zMrgb, zYrgb, zRrgb, zGrgb, zBrgb]
latentVecList = [zBlack, zWhite, zOrange, zPurple, zYellow, zRed, zGreen, zBlue]
zFinal = getLatent(startRGB, targetRGB, RGBvecList, zInput, latentVecList, zAvg)


############# Image generation and showing #############
#fakeAvgImage = generator(zBlue)
#saveImage(fakeAvgImage)

fakeImageWithFeature = generator(zFinal)
saveImage(fakeImageWithFeature)
