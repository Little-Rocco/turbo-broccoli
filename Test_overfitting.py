# NOTE: Before you can run it you need to make sure you have the below packages
#           - opencv (imported as cv2)
#               - using pip: https://www.tutorialspoint.com/how-to-install-opencv-in-python
#               - using conda: conda install -c conda-forge opencv
#           - scikit-image (imported as skimage)
#               - https://scikit-image.org/docs/dev/install.html

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os

#includes the functions used for calculating the
#structural similarity (structural simelarity index measure or SSIM) between two images
from skimage.metrics import structural_similarity as ssim

#includes mean squared error to measure difference in pixels
from skimage.metrics import mean_squared_error

#used for resizing
import cv2

#folder containing the generated images
generated_images = "C:\\Users\\fred7\\Documents\\GitHub\\turbo-broccoli\\test_images"

#folder containing the CelebA faces dataset
real_images = "C:\\Users\\fred7\\Documents\\GitHub\\turbo-broccoli\\test_images"

# values for checking the desired range of similarity between the real and generated images
checkM = 6000
checkS = 0.5

i=0

def compare(imageA, imageB, title):
    #resize images - only needed if the images have different dimensions
    res1 = cv2.resize(imageA, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(imageB, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)

    #calculate mean squared error
    m = mean_squared_error(res1, res2)

    #calculate structural similarity
    s = ssim(res1, res2, multichannel=True)

    # #im1 = cv2.imread(imageA, 0)
    # hist1 = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    #
    # #im2 = cv2.imread(imageB, 0)
    # hist2 = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    #
    # a = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    #if m <= checkM or s >= checkS:
    #setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # plt.suptitle("MSE: %.2f, SSIM: %.2f, HIST: %.2f" % (m, s, a))

    #show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    #show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    #plt.savefig("C:\\Users\\fred7\\Documents\\GitHub\\turbo-broccoli\\Refactored\\sus images\\" + "sus_comparison" + str(i) + ".png")


from datetime import datetime

time1 = datetime.now()

#time1 = time1.strftime("%H:%M:%S")


#iterate over real and generated images
for file_name1 in os.listdir(real_images):
    for file_name2 in os.listdir(generated_images):

        #read generated images
        filepath1 = os.path.join(generated_images, file_name2)
        image01 = io.imread(filepath1)

        #read real images
        filepath2 = os.path.join(real_images, file_name1)
        image02 = io.imread(filepath2)

        #plt.show()

        #prints the below message for every comparison
        if i % 1000 == 0:
            print("I am still runnning :)", i, "comparisons")

        #compare the generated and real images
        compare(image01, image02, "generated image vs. real image")

        i = i + 1

time2 = datetime.now()

#time2 = time2.strftime("%H:%M:%S")

final_time = time2 - time1

print(final_time.total_seconds())
