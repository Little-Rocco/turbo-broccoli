# Documentation
# install for IPython: https://ipython.org/install.html 
# 
# 
# end of documentation

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

# Root directory for dataset
dataroot = "C:\\Users\\Silas Bachmann\\Downloads\\archive\\"
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
iters_between_updates = 80

# Number of iterations to wait before showing graphs
iters_between_each_graph = 696*1

# Number of epochs inbetween model saves
epochsPerSave = 1

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(image_size),
                                                torchvision.transforms.CenterCrop(image_size),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5), (0.5)),
                                            ]))

print("Enter the number of the model you want to load, else just press enter for training")
modeChoice = input()

# Divides the dataset in into train and test
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Used to run model without learning
print("Disable learning? (y/n)")
learningChoice = input()

if (learningChoice != 'y'):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    iters_per_epoch = (train_size/batch_size)
else: 
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    iters_per_epoch = (test_size/batch_size)

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
    
    discriminator_network.eval()
    generator_network.eval()
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
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(generator_losses_x, generator_losses, label="G")
            plt.plot(discriminator_losses_x, discriminator_losses, label="D")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            #plt.show()
            plt.savefig("images/plot" + str(epoch) + "_" + now.strftime("%d-%m-%y") + ".png", normalize = True)

            # %%capture
            # fig = plt.figure(figsize=(8, 8))
            # plt.axis("off")
            # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
            # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

            # HTML(ani.to_jshtml())
            # plt.show()

            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # Plot the real images
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                        (1, 2, 0)))

            # Plot the fake images from the last epoch
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(img_list[-1].to(device)[:64], padding=10, normalize=True).cpu(),
                        (1, 2, 0)))
            #plt.show()
            plt.savefig("images/" + str(epoch) + "_" + now.strftime("%d-%m-%y") + ".png", normalize = True)

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