"""ATORpt_dataLoader.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt dataLoader

"""

import torch as pt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ATORpt_globalDefs import *

if(databaseName == "ALOI-VIEW"):
		
	class ALOIVIEWDataset(Dataset):
		def __init__(self, train=True):
			self.train = train
			
		def __len__(self):
			if(debugProcessSingleImage):
				return 1
			else:
				if(self.train):
					return (ALOIdatabaseNumberOfImages*ALOIdatabaseNumberOfViewsTrain)
				else:
					return (ALOIdatabaseNumberOfImages*ALOIdatabaseNumberOfViewsTest)

		def __getitem__(self, idx):
			if(debugProcessSingleImage):
				if(self.train):
					imageIndex = debugProcessSingleImageIndexTrain
					viewIndex = debugProcessSingleViewIndexTrain
				else:
					imageIndex = debugProcessSingleImageIndexTest
					viewIndex = debugProcessSingleViewIndexTest
			else:
				if(self.train):
					imageIndex = (idx//ALOIdatabaseNumberOfViewsTrain)
					viewIndex = idx%ALOIdatabaseNumberOfViewsTrain
				else:
					imageIndex = (idx//ALOIdatabaseNumberOfViewsTest)
					viewIndex = idx%ALOIdatabaseNumberOfViewsTest + ALOIdatabaseNumberOfViewsTrain
			return imageIndex, viewIndex
						
	def createDataloader():
		trainDataset = ALOIVIEWDataset(train=True)	#transform=transform
		testDataset = ALOIVIEWDataset(train=False)	#transform=transform
		trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=databaseTrainShuffle, generator=pt.Generator(device='cuda'))
		testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, generator=pt.Generator(device='cuda'))
		return trainDataLoader, testDataLoader

	def loadALOIVIEWimage(imageIndex, viewIndex):
		imagePath = getALOIVIEWImagePath(imageIndex, viewIndex)
		image = Image.open(imagePath)
		#image = image.convert("RGB")
		transform = transforms.ToTensor()
		imageTensor = transform(image)
		if(batchSize == 1):
			imagesTensor = pt.unsqueeze(imageTensor, dim=0)
		else:
			printe("loadALOIVIEWimage error: currently requires (batchSize == 1)")
		return imagesTensor
		
	def getALOIVIEWImagePath(imageIndices, viewIndices):
		imagePathList = []
		#print("imageIndices.shape = ", imageIndices.shape)
		for i in range(imageIndices.shape[0]):
			imageIndex = imageIndices[i].item()+ALOIdatabaseImageStartIndex
			viewIndex = viewIndices[i].item()
			viewName = str(int(viewIndex*(360/ALOIdatabaseNumberOfViews)))
			imageName = str(imageIndex) + "_r" + viewName + ".png"
			imagePath = databaseRoot + "aloi_view/png/" + str(imageIndex) + "/" + imageName
			imagePathList.append(imagePath)
		return imagePathList
			
elif(databaseName == "MNIST"):
	from torchvision.datasets.mnist import MNIST

	def createDataloader():
		trainDataset = MNIST(root='./../datasets', train=True, download=True, transform=transforms.ToTensor())
		testDataset = MNIST(root='./../datasets', train=False, download=True, transform=transforms.ToTensor())
		trainDataLoader = DataLoader(trainDataset, shuffle=databaseTrainShuffle, batch_size=batchSize, generator=pt.Generator(device='cuda'))
		testDataLoader = DataLoader(testDataset, shuffle=False, batch_size=batchSize, generator=pt.Generator(device='cuda'))
		return trainDataLoader, testDataLoader
		

