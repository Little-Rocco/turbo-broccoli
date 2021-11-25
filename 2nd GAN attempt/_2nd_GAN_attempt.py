# Documentation
# install for IPython: https://ipython.org/install.html 
# "hack" to allow for loading sibling packages: https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944
# 
# end of documentation


#the hack for loading sibling packages
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"

from Utility.graphHelper import *
from Utility.dataHelper import *
from Utility.Engine import *

#reset path and package
path.pop()
__package__ = None



from pickle import FALSE
import torch
import torchvision.models as models
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from datetime import datetime
from IPython.display import HTML

# --Use this if you struggle with figuring out where models and images are saved--
# import os
# cwd = os.getcwd()
# print("Current working directory: {0}".format(cwd))

# Root directory for dataset
dataroot = "C:\\Users\\Anders\\source\\repos\\data\\celeba"
# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
colour_channels = 3

# Size of z latent vector (i.e. size of generator input)
gen_input_nodes = 100

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers - no touching!
learning_rate = 0.0002

# Beta1 hyperparam for Adam optimizers - no touching!
beta1_hyperparam = 0.5

# Number of iterations to wait before printing updates
iters_between_updates = 50

# Number of iterations to wait before showing graphs
iters_between_each_graph = 100 #696*1

# Number of epochs inbetween model saves
epochsPerSave = 1



# initializes a seed
seed = torch.Generator().seed()
print("Current seed: " + str(seed))

# Loads and divides the dataset in into train and test
dataset = dataLoad.loadImages(dataroot, image_size)
train_dataset, test_dataset = dataLoad.splitData(dataset, 0.7, seed)

# Used to load models
print("Enter the number of the model you want to load, else just press enter for training")
modeChoice = input()

# Used to run model without learning
print("Disable learning? (y/n)")
learningChoice = input()
if (learningChoice != 'y'):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
else: 
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
iters_per_epoch = (math.ceil(len(dataloader.dataset)/batch_size))


# Decide which device we want to run on
device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda:0"

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            # the neural network
            torch.nn.Linear(gen_input_nodes, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, image_size * image_size * colour_channels),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x).reshape((x.shape[0], colour_channels, image_size, image_size))
        return logits
        
generator_network = Generator().to(device)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(image_size * image_size * colour_channels, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

discriminator_network = Discriminator().to(device)

loss_function = torch.nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, gen_input_nodes, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label  = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizer_discriminator = torch.optim.Adam(discriminator_network.parameters(), lr=learning_rate,
                                           betas=(beta1_hyperparam, 0.999))
optimizer_generator = torch.optim.Adam(generator_network.parameters(), lr=learning_rate,
                                       betas=(beta1_hyperparam, 0.999))


# Lists to keep track of progress

img_list = []
generator_losses = []
generator_losses_x = []
discriminator_losses = []
discriminator_losses_x = []
iterations = 0
epoch = 0


# Loads a saved checkpoint
if (modeChoice != ''):
    model = torch.load('Models\\Model' + modeChoice + '.pth')
    
    discriminator_network.load_state_dict(model['Discriminator'])
    optimizer_discriminator.load_state_dict(model['DiscriminatorOptimizer'])
    discriminator_losses = model['DiscriminatorLosses']
    discriminator_losses_x = model['DiscriminatorLosses_x']

    generator_network.load_state_dict(model['Generator'])
    optimizer_generator.load_state_dict(model['GeneratorOptimizer'])
    generator_losses = model['GeneratorLosses']
    generator_losses_x = model['GeneratorLosses_x']

    epoch = model['Epoch']
    iterations = model['Iterations']
    img_list = model['GenImgs']
    
    discriminator_network.eval()
    generator_network.eval()

    # Uncomment to find the seed used in a given model
    #seed = model['Seed']
    #print("Seed used on this epoch: " + str(seed))
    
    print("\n model #" + modeChoice + " loaded")


# -------- Training Loop ----------
print("Starting Loop...")
for epoch in range(epoch, num_epochs+1):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        discriminator_network.zero_grad()

        # Format batch
        real_image = data[0].to(device)
        real_batch_size = real_image.size(0)
        label = torch.full((real_batch_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through Discriminator
        output = discriminator_network(real_image).view(-1)

        # Calculate loss on all-real batch
        discriminator_loss_real = loss_function(output, label)

        # Calculate gradients for Discriminator in backward pass
        discriminator_loss_real.backward()

        discriminator_real_input_confidence = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(real_batch_size, gen_input_nodes, 1, 1, device=device)

        # Generate fake image batch with Generator
        fake = generator_network(noise)
        label.fill_(fake_label)

        # Classify all fake batch with Discriminator
        output = discriminator_network(fake.detach()).view(-1)

        # Calculate Discriminator's loss on the all-fake batch
        discriminator_loss_fake = loss_function(output, label)
    
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        discriminator_loss_fake.backward()
        discriminator_fake_input_confidence_1 = output.mean().item()

        # Compute error of Discriminator as sum over the fake and the real batches
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake

        if (learningChoice != 'y'):
            # Update Discriminator
            optimizer_discriminator.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        generator_network.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated Discriminator, perform another forward pass of all-fake batch through Discriminator
        output = discriminator_network(fake).view(-1)

        # Calculate G's loss based on this output
        generator_loss = loss_function(output, label)

        # Calculate gradients for G
        generator_loss.backward()
        discriminator_fake_input_confidence_2 = output.mean().item()

        if (learningChoice != 'y'):
            # Update G
            optimizer_generator.step()

        # Output training/progress stats
        if i % iters_between_updates == 0 and learningChoice != 'y':
            print('[%5d/%5d][%5d/%5d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs,
                    i, len(dataloader),
                    discriminator_loss.item(),
                    generator_loss.item(),
                    discriminator_real_input_confidence,
                    discriminator_fake_input_confidence_1,
                    discriminator_fake_input_confidence_2))
            # For time stamps
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S | %d-%m-%y")
            print("Current Time =", current_time)
            

        elif i % iters_between_updates == 0:
            print('[%5d/%5d][%5d/%5d]'
                % (epoch, num_epochs, i, len(dataloader),
                    ))
            # For time stamps
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S | %d-%m-%y")
            print("Current Time =", current_time)


        # Save Losses for plotting later
        generator_losses.append(generator_loss.item())
        generator_losses_x.append(iterations / iters_per_epoch)
        discriminator_losses.append(discriminator_loss.item())
        discriminator_losses_x.append(iterations / iters_per_epoch)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = generator_network(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))        

        # Show graphs
        if (iterations % iters_between_each_graph == iters_between_each_graph-1) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            
            #save graph for loss
            g_loss_curve = curve(generator_losses_x, generator_losses, "G")
            d_loss_curve = curve(discriminator_losses_x, discriminator_losses, "D")
            loss_graph = graph([g_loss_curve, d_loss_curve], "Epochs", "Loss", "Generator and Discriminator Loss During Training")
            saver.saveGraph(loss_graph,
                            directory="images",
                            filename="plot" + str(iterations) + "_" + now.strftime("%d-%m-%y") + ".png")


            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # save the images
            saver.saveImages(real_batch, img_list,
                             directory="images", 
                             filename=str(iterations) + "_" + now.strftime("%d-%m-%y") + ".png",
                             device=device)

        iterations += 1

    # Saves the model and more
    if(epoch % epochsPerSave == 0):
        #torch.save(model.state_dict(), 'model_weights.pth' + str(i))
        torch.save({
        'Discriminator': discriminator_network.state_dict(),
        'DiscriminatorOptimizer': optimizer_discriminator.state_dict(),
        'DiscriminatorLosses': discriminator_losses,
        'DiscriminatorLosses_x': discriminator_losses_x,

        'Generator': generator_network.state_dict(),
        'GeneratorOptimizer': optimizer_generator.state_dict(),
        'GeneratorLosses': generator_losses,
        'GeneratorLosses_x': generator_losses_x,

        'Epoch': epoch+1,
        'Iterations': iterations,
		'GenImgs': img_list,

        'Seed': seed,

        }, 'Models\\Model' + str(epoch) + '.pth')