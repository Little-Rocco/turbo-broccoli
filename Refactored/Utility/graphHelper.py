# Documentation
# Plotting modified from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 
# Memory leak fix after imports from: https://github.com/matplotlib/matplotlib/issues/20490 
# 
# 
# end of documentation

import os
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


class curve:
    """the set of x, y coordinates for a single curve in a graph"""
    xCoords = []
    yCoords = []
    label = ""
    def __init__(self, _xCoords, _yCoords, _label):
        self.xCoords = _xCoords
        self.yCoords = _yCoords
        self.label = _label

class graph:
    """a set of curves for graphing"""
    curves = []
    xLabel = ""
    yLabel = ""
    title = ""
    def __init__(self, _curves, _xLabel, _yLabel, _title):
        self.curves = _curves
        self.xLabel = _xLabel
        self.yLabel = _yLabel
        self.title = _title

class saver:
    """function(s) for plotting graphs"""
    def saveGraph(graph, directory, filename):
        fname = directory + os.path.sep + filename
        plt.figure(figsize=(10, 5))
        plt.title(graph.title)
        for c in graph.curves:
            plt.plot(c.xCoords, c.yCoords, label=c.label)
        plt.xlabel(graph.xLabel)
        plt.ylabel(graph.yLabel)
        plt.legend()
        plt.savefig(fname)
        plt.close()


    def saveImages(realImgs, fakeImgs, directory, filename, device):
        fname = directory + os.path.sep + filename

        #plot real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(torchvision.utils.make_grid(realImgs[0].to(device)[:64], padding=5, normalize=True).cpu(),
                    (1, 2, 0)))

        #plot fake images
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(
            np.transpose(torchvision.utils.make_grid(fakeImgs[:64], padding=5, normalize=True).cpu(),
                    (1, 2, 0)))

        plt.savefig(fname)
        plt.close()


    def saveImage(img, directory, filename, device, channels):
        fname = directory + os.path.sep + filename

        plt.figure(figsize=(1, 1), dpi=83)
        img_0 = img.to(device)
        if(channels == 1):
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(img_0[0], padding=0, normalize=True).cpu(),
                        (0, 1)))
        else:
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(img_0, padding=0, normalize=True).cpu(),
                        (1, 2, 0)))
        plt.savefig(fname)
        plt.close()

