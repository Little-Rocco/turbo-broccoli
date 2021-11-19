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

#reset path and package
path.pop()
__package__ = None



import torch
import torch.cuda
import torchvision.models as models
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from datetime import datetime
from IPython.display import HTML

# Root directory for dataset
dataroot = "C:\\Users\\Anders\\source\\repos\\data\\Fruits_360\\Training"
# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

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

# Number of epochs to wait before showing graphs
iters_between_each_graph = 100 #480

# Number of epochs inbetween model saves
epochsPerSave = 1

dataset = dataLoad.loadImages(dataroot, image_size)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
iters_per_epoch = (math.ceil(len(dataloader.dataset.imgs)/batch_size))

# Decide which device we want to run on
device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda:0"


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x).reshape((x.shape[0], colour_channels, image_size, image_size))
        return logits


generator_network = Generator().to(device)


# print(generator_network)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


discriminator_network = Discriminator().to(device)
# print(discriminator_network)


loss_function = torch.nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, gen_input_nodes, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
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
epochs = 0

# Loads a saved checkpoint
print("Enter the number of the model you want to load, else just press enter for training")
modeChoice = input()

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
    
    discriminator_network.eval()
    generator_network.eval()
    print("\n model #" + modeChoice + " loaded")

# Used to run model without learning
print("Disable learning? (y/n)")
learningChoice = input()

# -------- Training Loop ----------

print("Starting Training Loop...")

for epoch in range(epochs, num_epochs+1):
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

        # Output training stats
        if i % iters_between_updates == 0 and learningChoice != 'y':
            print('[%5d/%5d][%5d/%5d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs,
                     i, len(dataloader),
                     discriminator_loss.item(),
                     generator_loss.item(),
                     discriminator_real_input_confidence,
                     discriminator_fake_input_confidence_1,
                     discriminator_fake_input_confidence_2,))
            # For time stamps
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
        elif i % iters_between_updates == 0:
            print('[%5d/%5d][%5d/%5d]'
                % (epoch, num_epochs, i, len(dataloader),
                    ))
            # For time stamps
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
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

        }, 'Models\\Model' + str(epoch) + '.pth')
