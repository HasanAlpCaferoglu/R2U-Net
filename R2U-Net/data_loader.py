import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class OxfordPetDataset(data.Dataset):
	def __init__(self, root,image_size=256,mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob

		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
		

	def __getitem__(self, idx):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[idx]
		filename = ''.join((image_path.split('/')[-1]).split(' ')[-1:])[:-len(".jpg")] # this statement is ours
		# filename = image_path.split('_')[-1][:-len(".jpg")] # this line from original code
		GT_path = self.GT_paths + filename + '.png'

		image = Image.open(image_path)
		image = image.convert('RGB')
		
		GT = Image.open(GT_path)

		# aspect_ratio = image.size[1]/image.size[0]
		aspect_ratio = 1

		Transform = []

		#ResizeRange = random.randint(300,320)
		#Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
		Transform.append(T.Resize((256,256)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		
		image = Transform(image)
		GT = Transform(GT)

		try:
			Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			image = Norm_(image)
		except Exception as e:
			print("***************PATH PATH!!!!----------------- ", image_path, "error: ", e)
		return image, GT, filename

	def __len__(self):
		"""Returns the total number of image files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	print("----1")
	dataset =OxfordPetDataset(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
