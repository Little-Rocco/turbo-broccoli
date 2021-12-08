from posixpath import curdir
from numpy.lib.function_base import append
import torch
import os

from torch.autograd import variable

#too many indices for tensor of dim 4

path = 'Refactored' + os.path.sep + 'LatentSpace' + os.path.sep
i = 0
latentSpaceAverage = torch.zeros((64, 100, 1, 1))


x1 = os.walk(path)
x2 = list(x1)
x3 = x2[0]
fileNames = x3[2]
for fName in fileNames:
   latentSpace = torch.load(path +  os.path.sep + fName)['latentSpace']
   torch.isnan(latentSpace)
   latentSpaceAverage += latentSpace
   i += 1



latentSpaceAverage = latentSpaceAverage/i
print('this is some average i hope: \n' + str(latentSpaceAverage))