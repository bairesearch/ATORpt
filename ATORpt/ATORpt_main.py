"""ATORpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and pytorch 1.7+

conda create -n pytorchenv
source activate pytorchenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tqdm
conda install matplotlib

# Usage:
source activate pytorchenv
python3 ATORpt_main.py

# Description:
ATORpt main - train Axis Transformation Object Recognition neural network (ATOR)

ATORpt is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- ATORpt contains various modules for an end-to-end neural model of ATOR
- ATORpt is designed to perform transformations of all image pixel coordinates in parallel
- architecture layout provided (implementation incomplete)
- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
- supports classification of 2D image snapshots recreated from transformed mesh coordinates - standard
	- perform independent, parallelised target prediction of object triangle data
- supports feature detection via a CNN
- currently supports 2DOD (2D/image input object data)
- currently uses MNIST dataset to test affine(/euclidean/projective) invariance
- also supports useMultKeys - modify transformer to support geometric hashing operations - experimental

# Future:
Requires upgrading to support 3DOD (3D input object data)

"""

import torch as pt
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from tqdm import tqdm

from ATORpt_globalDefs import *
import ATORpt_ATOR
import ATORpt_vit
import ATORpt_operations

def main():

	#template: https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py
	
	trainDataset = MNIST(root='./../datasets', train=True, download=True, transform=ToTensor())
	testDataset = MNIST(root='./../datasets', train=False, download=True, transform=ToTensor())

	batchSize = 16 #debug: 2
	inputShape = (1, 28, 28)   #MNIST defined	#numberOfChannels, imageHeight, imageWidth
	numberOfOutputDimensions = 10	 #MNIST defined
	
	trainDataLoader = DataLoader(trainDataset, shuffle=True, batch_size=batchSize, generator=pt.Generator(device='cuda'))
	testDataLoader = DataLoader(testDataset, shuffle=False, batch_size=batchSize, generator=pt.Generator(device='cuda'))
	
	numberOfPatchesATOR = inputShape[1]
	patchSizeATOR = ATORpt_operations.getPatchSize(inputShape, numberOfPatchesATOR)

	if(useClassificationVIT):
		if(useParallelisedGeometricHashing):
			numberOfPatchesVIT = numberOfPatchesATOR
		else:
			numberOfPatchesVIT = 7
		patchSizeVIT = ATORpt_operations.getPatchSize(inputShape, numberOfPatchesVIT)
		numberOfInputDimensionsVIT = ATORpt_operations.getInputDim(inputShape, patchSizeVIT)
		if(useParallelisedGeometricHashing):
			numberOfHiddenDimensionsVIT = numberOfInputDimensionsVIT + numberOfGeometricDimensions   #mandatory (currently required for ATORpt_operations.getPositionalEmbeddingsAbsolute, as no method implemented for mapping between hashing_d and numberOfHiddenDimensionsVIT)
		else:
			numberOfHiddenDimensionsVIT = 20   #arbitrary
		numberOfHeadsVIT = 1	#requirement; numberOfHiddenDimensionsVIT % numberOfHeadsVIT == 0

	device = pt.device('cuda')
	
	modelParameters = []
	if(useParallelisedGeometricHashing):
		if(not positionalEmbeddingTransformationOnly):
			ATORmodel = ATORpt_ATOR.ATORmodelClass(inputShape, numberOfPatchesATOR)
			modelParameters = modelParameters + list(ATORmodel.parameters()) 
		
	if(useClassificationSnapshots):
		pass
	elif(useClassificationVIT):
		VITmodel = ATORpt_vit.ViTClass(inputShape, numberOfPatchesVIT, numberOfHiddenDimensionsVIT, numberOfHeadsVIT, numberOfOutputDimensions)
		modelParameters = modelParameters + list(VITmodel.parameters()) 

	numberOfEpochs = 10
	learningRate = 0.05

	optimizer = Adam(modelParameters, lr=learningRate)
	criterion = CrossEntropyLoss()
	for epoch in tqdm(range(numberOfEpochs), desc="Train"):
		trainLoss = 0.0
		for batch in tqdm(trainDataLoader, desc=f"Epoch {epoch + 1}", leave=False):
			x, y = batch
			x, y = x.to(device), y.to(device)
			
			imagesOrig = x
			batchSize, numberOfChannels, imageHeightOrig, imageWidthOrig = imagesOrig.shape
			for r in range(0, numberOfSpatialResolutions):
				if(useMultipleSpatialResolutions):
					spatialResolution = 2**r		
					spatialResolutionTransform = T.Resize((imageHeightOrig//spatialResolution, imageWidthOrig//spatialResolution))
					images = spatialResolutionTransform(imagesOrig)
				else:
					images = imagesOrig
			
				if(useParallelisedGeometricHashing):
					if(positionalEmbeddingTransformationOnly):
						posEmbeddingsAbsoluteGeoNormalised = ATORpt_operations.getPositionalEmbeddingsAbsolute(numberOfPatchesVIT)	#or numberOfPatchesATOR
						posEmbeddingsAbsoluteGeoNormalised = pt.unsqueeze(posEmbeddingsAbsoluteGeoNormalised, dim=0)
						posEmbeddingsAbsoluteGeoNormalised = posEmbeddingsAbsoluteGeoNormalised.repeat(batchSize, 1, 1)
					else:
						posEmbeddingsAbsoluteGeoNormalised = ATORmodel(images)
				else:
					posEmbeddingsAbsoluteGeoNormalised = None
					
				if(useClassificationSnapshots):
					#recreate a 2D image from the transformed mesh before performing object recognition
						#perform independent/parallised target prediction of object triangle data
					print("warning: useClassificationSnapshots is incomplete")
				elif(useClassificationVIT):
					y_hat = VITmodel(images, posEmbeddingsAbsoluteGeoNormalised)


				loss = criterion(y_hat, y) / len(x)

				trainLoss += loss.detach().cpu().item()

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		print(f"Epoch {epoch + 1}/{numberOfEpochs} loss: {trainLoss:.2f}")

	correct, total = 0, 0
	testLoss = 0.0
	for batch in tqdm(testDataLoader, desc="Test"):
		x, y = batch
		x, y = x.to(device), y.to(device)
		
		imagesOrig = x
		batchSize, numberOfChannels, imageHeightOrig, imageWidthOrig = imagesOrig.shape
		for r in range(0, numberOfSpatialResolutions):
			if(useMultipleSpatialResolutions):
				spatialResolution = 2**r		
				spatialResolutionTransform = T.Resize((imageHeightOrig//spatialResolution, imageWidthOrig//spatialResolution))
				images = spatialResolutionTransform(imagesOrig)
			else:
				images = imagesOrig
	
			posEmbeddingsAbsoluteGeoNormalised = ATORmodel(images)

			if(useClassificationSnapshots):
				print("warning: useClassificationSnapshots is incomplete")	
			elif(useClassificationVIT):
				y_hat = VITmodel(images, posEmbeddingsAbsoluteGeoNormalised)
									
			loss = criterion(y_hat, y) / len(x)
			testLoss += loss.detach().cpu().item()

			correct += pt.sum(pt.argmax(y_hat, dim=1) == y).detach().cpu().item()
			total += len(x)

	print(f"Test loss: {testLoss:.2f}")
	print(f"Test accuracy: {correct / total * 100:.2f}%")



if __name__ == '__main__':
	main()
