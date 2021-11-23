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

		self.dataset = dataLoad.loadImages(dataroot, self.opt.img_size)
		# the outcommented is done in Engine.enter_info()
		#self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
		#self.iters_per_epoch = (math.ceil(len(dataloader.dataset.imgs)/opt.batch_size))
		self.cuda = True if torch.cuda.is_available() else False


	generator = None
	discriminator = None
	def add_networks(self, generator, discriminator):
		self.generator = generator
		self.discriminator = discriminator
		if self.cuda:
			self.generator.cuda()
			self.discriminator.cuda()


	optimizer_G = None
	optimizer_D = None
	loss_func = None
	def add_functions(self, optimizer_G, optimizer_D, loss_func):
		self.optimizer_G = optimizer_G
		self.optimizer_D = optimizer_D
		self.loss_func = loss_func


	gen_imgs = []
	real_losses = []
	generator_losses = []
	discriminator_losses = []
	losses_x = []
	epochs_done = 0
	iters_done = 0
	current_iter = 0
	accuracy_real, accuracy_fake = 0, 0
	fixed_noise = None
	Tensor = None
	def run(self):
		self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
		self.enter_info()
		self.fixed_noise = torch.randn(64, self.opt.latent_dim, 1, 1, device="cpu")
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

				# append x values
				self.losses_x.append(self.iters_done / self.iters_per_epoch)

				# save graphs (and images)
				if(i % self.opt.sample_interval == 0):
					self.save_graphs()

				self.iters_done += 1
				self.current_iter += 1
				self.gen_imgs = []

			if(epoch % self.opt.epochs_per_save == 0):
				if(self.learningChoice != 'y'):
					self.save_checkpoint()

			self.epochs_done += 1
			self.current_iter = 0


	#####################################
	#     Helper functions for run()    #
	#####################################

	modeChoice = ""
	learningChoice = ""
	iters_per_epoch = 0
	def enter_info(self):
		# Used to load models
		print("Enter the number of the model you want to load, else just press enter for training")
		self.modeChoice = input()

		# Used to run model without learning
		train_dataset, test_dataset = dataLoad.splitData(self.dataset, self.train_ratio, self.seed)
		print("Disable learning? (y/n)")
		self.learningChoice = input()
		if (self.learningChoice != 'y'):
			self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True)
		else: 
			self.dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.opt.batch_size, shuffle=True)
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

		self.epochs_done = model['Epoch']
		self.iters_done = model['Iterations']

		self.seed = model['Seed']
		
		self.discriminator.eval()
		self.generator.eval()

		# Uncomment to find the seed used in a given model
		#seed = model['Seed']
		#print("Seed used on this epoch: " + str(seed))
		
		print("\n model #" + self.modeChoice + " loaded")


	def save_checkpoint(self):
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

			'Epoch': self.epochs_done+1,
			'Iterations': self.iters_done,

			'Seed': self.seed,
        }, 
		'Models\\Model' + str(self.epochs_done) + '.pth')



	z = None
	real_batch_size = 0
	def train_discriminator(self, imgs):
		# Configure input
		real_imgs = Variable(imgs.type(self.Tensor))

		# get a label vector for true label
		self.real_batch_size = real_imgs.size(0)
		device = "cuda:0" if self.cuda else "cpu"
		labels = torch.full((self.real_batch_size, 1, 1, 1), 1, dtype=torch.float, device=device)

		# use real data
		self.optimizer_D.zero_grad()
		real_preds = self.discriminator(real_imgs)
		real_loss = self.loss_func(real_preds, labels)

		# use generated data
		self.z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.opt.latent_dim, 1, 1))))
		fake_imgs = self.generator(self.z).detach()
		labels2 = torch.full((self.real_batch_size, 1, 1, 1), 0, dtype=torch.float, device=device)
		fake_preds = self.discriminator(fake_imgs)
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
					
		self.real_losses.append(real_loss.item())
		self.discriminator_losses.append(loss_D.item())


	def use_clipping(self):
		return self.opt.clip_value != -1



	def train_generator(self):
		# fake labels are real for generator cost
		device = "cuda:0" if self.cuda else "cpu"
		labels = torch.full((self.real_batch_size, 1, 1, 1), 1, dtype=torch.float, device=device)

		# run generator
		self.optimizer_G.zero_grad()
		fake_imgs = self.generator(self.z)
		self.gen_imgs.append(fake_imgs)
		loss_G = self.loss_func(self.discriminator(fake_imgs), labels)

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
		g_loss_curve = curve(self.losses_x, self.generator_losses, "Generator Loss (D(G(z)))")
		d_loss_curve1 = curve(self.losses_x, self.discriminator_losses, "Critic Loss, Fake (D(x) - D(G(z)))")
		d_loss_curve2 = curve(self.losses_x, self.real_losses, "Critic Loss, Real (D(x))")
		loss_graph = graph([g_loss_curve, d_loss_curve1, d_loss_curve2], "Epochs", "Loss", "Generator and Discriminator Loss During Training")
		saver.saveGraph(loss_graph,
                        directory="images",
                        filename="plot_%d.png" % self.iters_done)

        # Grab a batch of real images from the dataloader
		real_batch = next(iter(self.dataloader))

        # save the images
		saver.saveImages(real_batch, self.gen_imgs,
                            directory="images", 
                            filename="%d.png" % self.iters_done,
                            device="cpu")
