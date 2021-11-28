#to-do: add a range for when we want to observe if two images are similar

# NOTE: Before you can run it you need to make sure you have the below packages
#           - opencv (imported as cv2)
#               - using pip: https://www.tutorialspoint.com/how-to-install-opencv-in-python
#               - using conda: conda install -c conda-forge opencv
#           - scikit-image (imported as skimage)
#               - https://scikit-image.org/docs/dev/install.html

#folder containing the generated images
generated_images = "C:\\Users\\fred7\\PycharmProjects\\turbo-broccoli\\2nd GAN attempt\\generated images"

#folder containing the CelebA faces dataset
real_images = "C:\\Users\\fred7\\Documents\\GitHub\\turbo-broccoli\\CelebA dataset\\img_align_celeba"

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os

#includes the functions used for calculating the
#structural similarity (structural simelarity index measure or SSIM) between two images
from skimage.metrics import structural_similarity as ssim

#used for resizing
import cv2

#calculates the mean squared error between two images - sum of the squared difference between the two images
def mean_squared_error(imageA, imageB):

    mse = np.square(imageA.astype("float") - imageB.astype("float"))
    mse = np.sum(mse)
    mse = np.mean(mse / float(imageA.shape[0] * imageB.shape[1]))

    return mse

def compare_images(imageA, imageB, title):
    #resize images - only needed if the images have different dimensions
    res1 = cv2.resize(imageA, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(imageB, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)

    #calculate mean squared error
    m = mean_squared_error(res1, res2)

    #calculate structural similarity
    s = ssim(res1, res2, multichannel=True)

    #setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    #show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    #show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show

#iterate over real and generated images
for file_name1 in os.listdir(real_images):
    for file_name2 in os.listdir(generated_images):

        #read generated images
        filepath1 = os.path.join(generated_images, file_name2)
        image01 = io.imread(filepath1)

        #read real images
        filepath2 = os.path.join(real_images, file_name1)
        image02 = io.imread(filepath2)

        #create a tuble that we can iterate through - here a title of the specific image is specified too
        fig = plt.figure("Images")
        images = ("generated image", image01), ("real_image", image02)

        #loop over the tuble containging the generated and real images
        for (i, (name, image)) in enumerate(images):

            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(name)
            plt.imshow(image, cmap = plt.cm.gray)
            plt.axis("off")
            plt.show()

            #compare the generated and real images
            compare_images(image01, image02, "generated image vs. real image")