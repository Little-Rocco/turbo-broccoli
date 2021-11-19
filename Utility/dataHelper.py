import torch
import torchvision

class dataLoad:
	"""functions to load data"""
	def loadImages(dataroot, image_size, grayscale=False):
		if(grayscale):
			dataset = torchvision.datasets.ImageFolder(root=dataroot,
											  transform=torchvision.transforms.Compose([
												  torchvision.transforms.grayscale(),
												  torchvision.transforms.Resize(image_size),
												  torchvision.transforms.CenterCrop(image_size),
												  torchvision.transforms.ToTensor(),
												  torchvision.transforms.Normalize((0.5), (0.5)),
												  ]))
		else:
			dataset = torchvision.datasets.ImageFolder(root=dataroot,
											  transform=torchvision.transforms.Compose([
												  torchvision.transforms.Resize(image_size),
												  torchvision.transforms.CenterCrop(image_size),
												  torchvision.transforms.ToTensor(),
												  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
												  ]))
		return dataset

	#split into training and test data
	def splitData(dataset, train_ratio, seed):
		train_size = int(train_ratio * len(dataset))
		test_size = len(dataset) - train_size
		train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
		return train_dataset, test_dataset
