from .graphHelper import *
from .dataHelper import *

import os
import argparse
import math
import torch

import numpy as np

from torch.autograd import Variable
from datetime import datetime



class Engine:
	opt = argparse.ArgumentParser()
	dataroot = ""
	dataset = None
	dataloader = None
	cuda = False
	seed = 0
	train_ratio = 0.0
	def __init__(self, options, dataroot, seed, train_ratio):
		self.opt = options
		self.dataroot = dataroot
		self.seed = seed
		self.train_ratio = train_ratio

		isGrayscale = self.opt.channels == 1
		self.dataset = dataLoad.loadImages(dataroot, self.opt.img_size, isGrayscale)
		# the outcommented is done in Engine.enter_info()
		#self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
		#self.iters_per_epoch = (math.ceil(len(dataloader.dataset.imgs)/opt.batch_size))
		self.enter_info()
		self.cuda = True if torch.cuda.is_available() else False


	generator = None
	discriminator = None
	def add_networks(self, generator, discriminator):
		device = "cuda:0" if self.cuda else "cpu"

		self.generator = generator
		self.discriminator = discriminator
		if self.cuda:
			self.generator.to(device)
			self.discriminator.to(device)


	optimizer_G = None
	optimizer_D = None
	loss_func = None
	def add_functions(self, optimizer_G, optimizer_D, loss_func):
		self.optimizer_G = optimizer_G
		self.optimizer_D = optimizer_D
		self.loss_func = loss_func


	gen_imgs = []
	real_losses = []
	fake_losses = []
	generator_losses = []
	discriminator_losses = []
	losses_x = []
	epochs_done = 0
	iters_done = 0
	current_iter = 0
	fixed_noise = []
	Tensor = None
	def run(self):
		self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
		
		for x in range(64):
			self.fixed_noise.append(torch.randn(64, self.opt.latent_dim, 1, 1, device="cpu"))

		if (self.modeChoice != ''):
			self.load_checkpoint()

		# -------- Training Loop ----------
		print("Starting Loop...")
		for epoch in range(self.epochs_done, self.opt.n_epochs+1):
			for i, (imgs, _) in enumerate(self.dataloader):
				# train discriminator
				self.train_discriminator(imgs)

				# only train generator every n_critic iterations
				if(i % self.opt.n_critic == 0):
					self.train_generator()

				# print to terminal
				if(i % self.opt.update_interval == 0):
					self.print_update()
					#print(self.optimizer_D.param_groups[0]['betas'])

				# append x values
				self.losses_x.append(self.iters_done / self.iters_per_epoch)

				# save graphs (and images)
				if(i % self.opt.sample_interval == 0):
					self.save_graphs()

				self.iters_done += 1
				self.current_iter += 1
				self.gen_imgs = []
				
			self.epochs_done += 1
			self.current_iter = 0

			# Model saving
			if(epoch % self.opt.epochs_per_save == self.opt.epochs_per_save-1):
				if(self.learningChoice != 'y'):
					self.save_checkpoint()



	#####################################
	#     Helper functions for run()    #
	#####################################

	modeChoice = ""
	learningChoice = ""
	individualImagesChoice = ""
	iters_per_epoch = 0
	def enter_info(self):
		train_dataset, test_dataset = dataLoad.splitData(self.dataset, self.train_ratio, self.seed)

		# Used to load models
		print("Enter the number of the model you want to load, else just press enter for training")
		self.modeChoice = input()

		# Used to run model without learning
		print("Disable learning? (y/n)")
		self.learningChoice = input()
		if (self.learningChoice != 'y'):
			self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True)
		else: 
			self.dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.opt.batch_size, shuffle=True)
			print("    Save individual images? (y/n)")
			self.individualImagesChoice = input()
		self.iters_per_epoch = (math.ceil(len(self.dataloader.dataset)/self.opt.batch_size))



	def load_checkpoint(self):
		model = torch.load('Models' + os.path.sep + 'Model' + self.modeChoice + '.pth')
		
		self.discriminator.load_state_dict(model['Discriminator'])
		self.optimizer_D.load_state_dict(model['DiscriminatorOptimizer'])
		self.discriminator_losses = model['DiscriminatorLosses']
		self.real_losses = model['RealLosses']

		self.generator.load_state_dict(model['Generator'])
		self.optimizer_G.load_state_dict(model['GeneratorOptimizer'])
		self.generator_losses = model['GeneratorLosses']

		self.losses_x = model['Losses_x']
		self.gen_imgs = model['GenImgs']

		self.epochs_done = model['Epoch']
		self.iters_done = model['Iterations']

		self.seed = model['Seed']
		self.fixed_noise = model['FixedNoise']
		
		if(self.learningChoice != 'y'):
			self.discriminator.train()
			self.generator.train()
		else:
			self.discriminator.eval()
			self.generator.eval()

		device = "cuda:0" if self.cuda else "cpu"
		self.discriminator.to(device)
		self.generator.to(device)

		# Uncomment to find the seed used in a given model
		#print("Seed used on this epoch: " + str(self.seed))
		
		print("\n model #" + self.modeChoice + " loaded")


	def save_checkpoint(self):
		device = "cuda:0" if self.cuda else "cpu"
		if(self.cuda):
			self.discriminator.to("cpu")
			self.generator.to("cpu")

        #torch.save(model.state_dict(), 'model_weights.pth' + str(i))
		torch.save({
			'Discriminator': self.discriminator.state_dict(),
			'DiscriminatorOptimizer': self.optimizer_D.state_dict(),
			'DiscriminatorLosses': self.discriminator_losses,
			'RealLosses': self.real_losses,

			'Generator': self.generator.state_dict(),
			'GeneratorOptimizer': self.optimizer_G.state_dict(),
			'GeneratorLosses': self.generator_losses,

			'Losses_x': self.losses_x,
			'GenImgs': self.gen_imgs,

			'Epoch': self.epochs_done,
			'Iterations': self.iters_done,

			'Seed': self.seed,
			'FixedNoise': self.fixed_noise,
        }, 
		'Models\\Model' + str(self.epochs_done) + '.pth')
		
		if(self.cuda):
			self.discriminator.to(device)
			self.generator.to(device)


	z = None
	real_batch_size = 0
	def train_discriminator(self, imgs):
		self.optimizer_D.zero_grad()

		# Configure input
		real_imgs = Variable(imgs.type(self.Tensor))

		# get a label vector for true label
		self.real_batch_size = real_imgs.size(0)
		device = "cuda:0" if self.cuda else "cpu"
		labels = torch.full((self.real_batch_size,), 1, dtype=torch.float, device=device)

		# use real data
		real_preds = self.discriminator(real_imgs).view(-1)
		real_loss = self.loss_func(real_preds, labels)

		# use generated data
		self.z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.opt.latent_dim, 1, 1))))
		fake_imgs = self.generator(self.z)
		labels2 = torch.full((self.real_batch_size,), 0, dtype=torch.float, device=device)
		fake_preds = self.discriminator(fake_imgs.detach()).view(-1)
		fake_loss = self.loss_func(fake_preds, labels2)

		# combine
		loss_D = real_loss + fake_loss

		# optimize
		if(self.learningChoice != 'y'):
			loss_D.backward()
			self.optimizer_D.step()
			if(self.use_clipping()):
				for p in self.discriminator.parameters():
					p.data.clamp_(-self.opt.clip_value, self.opt.clip_value)
					
		self.gen_imgs.append(fake_imgs)
		self.real_losses.append(real_loss.item())
		self.fake_losses.append(fake_loss.item())
		self.discriminator_losses.append(loss_D.item())


	def use_clipping(self):
		return self.opt.clip_value != -1



	def train_generator(self):
		self.optimizer_G.zero_grad()

		# fake labels are real for generator cost
		device = "cuda:0" if self.cuda else "cpu"
		labels = torch.full((self.real_batch_size,), 1, dtype=torch.float, device=device)

		# run generator
		fake_imgs = self.gen_imgs[0]
		output = self.discriminator(fake_imgs).view(-1)
		loss_G = self.loss_func(output, labels)

		# optimize
		if(self.learningChoice != 'y'):
			loss_G.backward()
			self.optimizer_G.step()

		self.generator_losses.append(loss_G.item())



	def print_update(self):
		print('[%5d/%5d][%5d/%5d]\tD(x): %.4f\tCritic loss: %.4f\tGenerator loss: %.4f\t'
            % (self.epochs_done, self.opt.n_epochs,
                self.current_iter, len(self.dataloader),
                self.real_losses[-1],
                self.discriminator_losses[-1],
                self.generator_losses[-1],))
        # For time stamps
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)



	def save_graphs(self):
        #save graph for loss
		loss_graph = None

		if(self.opt.split_disc_loss):
			g_loss_curve = curve(self.losses_x, self.generator_losses, "Generator Loss (D(G(z)))")
			d_loss_curve2 = curve(self.losses_x, self.real_losses, "Critic Loss, Real (D(x))")
			d_loss_curve1 = curve(self.losses_x, self.discriminator_losses, "Critic Loss, Fake (D(x) - D(G(z)))")
			loss_graph = graph([g_loss_curve, d_loss_curve1, d_loss_curve2], "Epochs", "Loss", "Generator and Discriminator Loss During Training")
		else:
			g_loss_curve = curve(self.losses_x, self.generator_losses, "G")
			d_loss_curve = curve(self.losses_x, self.discriminator_losses, "D")
			loss_graph = graph([g_loss_curve, d_loss_curve], "Epochs", "Loss", "Generator and Discriminator Loss During Training")
			
		saver.saveGraph(loss_graph,
                        directory="images",
                        filename="plot_%d.png" % self.iters_done)

        # Grab a batch of real images from the dataloader
		real_batch = next(iter(self.dataloader))

		savedImagesList = []
		device = "cuda:0" if self.cuda else "cpu"
		for x in range(64):
			savedImagesList.append(self.generator(self.fixed_noise[x].to(device))[0])


		if(self.individualImagesChoice != "y"):
			# save the images
			saver.saveImages(
								real_batch, savedImagesList,
								directory="images", 
								filename = str(self.epochs_done) + "e_" + str(self.iters_done) + "i.png",
								device="cpu")
		else:
			# save real images
			i = 0
			for img in real_batch[0]:
				saver.saveImage(
							img,
							directory="images" + os.path.sep + "individual_real",
							filename = str(self.epochs_done) + "e_" + str(self.iters_done) + "i_" + str(i) + "real.png",
							device="cpu")
				i += 1

			# save fake images
			i = 0
			for img in savedImagesList:
				saver.saveImage(
							img.detach(),
							directory="images" + os.path.sep + "individual_fake",
							filename = str(self.epochs_done) + "e_" + str(self.iters_done) + "i_" + str(i) + "fake.png",
							device="cpu")
				i += 1
