from posixpath import curdir
from numpy.lib.function_base import append
import torch
import os

from torch.autograd import variable

#MemoryLeak, not sure if this works

path = 'Refactored' + os.path.sep + 'LatentSpace' + os.path.sep
i = 0
j = 0
latentSpaces = []
latentSpaceAverage = 0


for i in range(1000):
   if os.path.isfile(path + str(i) + 'LS.pth'):
      latentSpaces.append(torch.load(path + str(i) + 'LS.pth'))
      print(j)
      print(latentSpaces[j])
      j += 1

for i in range(j):
   latentSpaceAverage += latentSpaces[i]

latentSpaceAverage = latentSpaceAverage/j

print(latentSpaceAverage)