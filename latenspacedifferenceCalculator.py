from posixpath import curdir
from numpy.lib.function_base import append
import torch
import os

from torch.autograd import variable

#MemoryLeak, not sure if this works

path = 'Refactored' + os.path.sep + 'LatentSpace' + os.path.sep
i = 0
latentSpaceAverage = torch.empty((2, 100, 1, 1))
latentSpaceAverage.detach()


x1 = os.walk(path)
x2 = list(x1)
x3 = x2[0]
fileNames = x3[2]
for fName in fileNames:
   latentSpace = torch.load(path + os.path.sep + fName)['latentSpace']
   latentSpace.detach()
   latentSpaceAverage += latentSpace
   i += 1



latentSpaceAverage = latentSpaceAverage/i
print(latentSpaceAverage)