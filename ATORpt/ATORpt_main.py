"""ATORpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and pytorch 2.2+

useATORCserialGeometricHashing:
	install all ATOR C++ prerequisites
	conda create --name pytorchsenv2 python=3.8
	source activate pytorchsenv2
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip3 install tqdm
	pip3 install transformers
	pip3 install click
	pip3 install opencv-python opencv-contrib-python
	pip3 install kornia
!useATORCserialGeometricHashing:
	conda create -n pytorchsenv
	source activate pytorchsenv
	conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
	conda install tqdm
	conda install transformers
	conda install matplotlib
	pip install opencv-python opencv-contrib-python
	pip install kornia

# Usage:
source activate pytorchsenv2
python3 ATORpt_main.py

# Description:
ATORpt main - train Axis Transformation Object Recognition neural network (ATOR)

ATORpt is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
- useATORCserialGeometricHashing:
	- uses ATOR C++ executable to generate transformed patches (normalised snapshots)
	- requires all ATOR C++ prerequisites 
- !useATORCserialGeometricHashing:
	- ATORpt contains various modules for an end-to-end neural model of ATOR
	- ATORpt is designed to perform transformations of all image pixel coordinates in parallel
	- architecture layout provided (implementation incomplete)
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
from tqdm import tqdm
from ATORpt_globalDefs import *
import ATORpt_dataLoader
if(useStandardVIT):
	import ATORpt_vitStandard
	import ATORpt_geometricHashingC
	import torch.optim as optim
	from torchvision import transforms, datasets
	from transformers import ViTFeatureExtractor, ViTForImageClassification
else:
	import ATORpt_vitCustom
	from torch.optim import Adam
	from torch.nn import CrossEntropyLoss
	import ATORpt_ATOR
	import ATORpt_operations

device = pt.device('cuda')

if(useStandardVIT):
	def mainStandardViT():
		print("mainStandardViT")

		trainDataLoader, testDataLoader = ATORpt_dataLoader.createDataloader()

		if(trainVITfromScratch):
			VITmodel = ATORpt_vitStandard.ViTForImageClassificationClass()
		else:
			feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
			config = feature_extractor.config
			config.num_labels = ALOIdatabaseNumberOfImages
			VITmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', config=config)

		trainOrTestStandardViT(True, trainDataLoader, VITmodel)
		trainOrTestStandardViT(False, testDataLoader, VITmodel)
else:
	def mainCustomViT():
		#template: https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py

		trainDataLoader, testDataLoader = ATORpt_dataLoader.createDataloader()
		inputShape = databaseImageShape

		#ATORpatchSize = ATORpt_operations.getPatchSize(inputShape, ATORnumberOfPatches)
		patchSizeVITlocal = ATORpt_operations.getPatchSize(inputShape, VITnumberOfPatches)

		if(useClassificationVIT):
			numberOfInputDimensionsVIT = ATORpt_operations.getInputDim(inputShape, patchSizeVITlocal)
			if(useParallelisedGeometricHashing):
				numberOfHiddenDimensionsVIT = numberOfInputDimensionsVIT + numberOfGeometricDimensions   #mandatory (currently required for ATORpt_operations.getPositionalEmbeddingsAbsolute, as no method implemented for mapping between hashing_d and numberOfHiddenDimensionsVIT)
			else:
				numberOfHiddenDimensionsVIT = 20   #arbitrary
			numberOfHeadsVIT = 1	#requirement; numberOfHiddenDimensionsVIT % numberOfHeadsVIT == 0

		modelParameters = []
		ATORmodel = None
		if(useParallelisedGeometricHashing):
			if(not positionalEmbeddingTransformationOnly):
				ATORmodel = ATORpt_ATOR.ATORmodelClass(inputShape, ATORnumberOfPatches)
				modelParameters = modelParameters + list(ATORmodel.parameters()) 
		if(useClassificationSnapshots):
			VITmodel = None
		if(useClassificationVIT):
			VITmodel = ATORpt_vitCustom.ViTClass(inputShape, VITnumberOfPatches, numberOfHiddenDimensionsVIT, numberOfHeadsVIT, numberOfOutputDimensions)
			modelParameters = modelParameters + list(VITmodel.parameters()) 

		trainOrTestCustomViT(True, trainDataLoader, ATORmodel, VITmodel, modelParameters)
		trainOrTestCustomViT(False, testDataLoader, ATORmodel, VITmodel, modelParameters)

if(useStandardVIT):
	def trainOrTestStandardViT(train, dataLoader, VITmodel):
		if(train):
			learningRate = 1e-4
			VITmodel.train()
			desc="Train"
			numberOfEpochs = trainNumberOfEpochs
			# Freeze all layers except the classification head
			for param in VITmodel.parameters():
				param.requires_grad = False
			for param in VITmodel.classification_head.parameters():
				param.requires_grad = True
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.Adam(VITmodel.parameters(), lr=learningRate)
		else:
			model.eval()
			correct = 0
			total = 0
			desc="Test"
			numberOfEpochs = 1

		for epoch in tqdm(range(numberOfEpochs), desc=desc):
			running_loss = 0.0
			correct = 0
			total = 0
			print("epoch = ", epoch)
			for batch in tqdm(dataLoader, desc=f"Epoch {epoch + 1}", leave=False):
				print("batch = ", batch)
				if(databaseName == "ALOI-VIEW"):
					imageIndex, viewIndex = batch
					labels = imageIndex
					imageIndex = imageIndex.item()
					viewIndex = viewIndex.item()
					imagePath = ATORpt_dataLoader.getALOIVIEWImagePath(imageIndex, viewIndex)
					transformedPatches = ATORpt_geometricHashingC.generateATORpatches(imagePath, train)	#normalisedsnapshots

				transformedPatches = transformedPatches.unsqueeze(0)	#add dummy batch size dimension (size 1)
				artificialInputImages = transformedPatches.view(batchSize, VITnumberOfChannels, VITimageSize, VITimageSize)
				print("labels = ", labels)
				print("transformedPatches.shape = ", transformedPatches.shape)
				if(train):
					optimizer.zero_grad()
					logits = VITmodel(artificialInputImages)
					loss = criterion(logits, labels)
					loss.backward()
					optimizer.step()
					running_loss += loss.item()
					_, predicted = pt.max(logits, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
				else:
					logits = VITmodel(artificialInputImages)
					_, predicted = pt.max(logits, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
			if(train):
				train_loss = running_loss / len(dataLoader)
				train_acc = correct / total
				print(f'Epoch [{epoch+1}/{numberOfEpochs}], Loss: {train_loss:.4f}, Accuracy: {100*train_acc:.2f}%')
		if(not train):
			test_acc = correct / total
			print(f'Test Accuracy: {100*test_acc:.2f}%')
else:
	def trainOrTestCustomViT(train, dataLoader, ATORmodel, VITmodel, modelParameters):
		#template: https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py

		if(train):
			learningRate = 0.05
			optimizer = Adam(modelParameters, lr=learningRate)
			criterion = CrossEntropyLoss()
			desc="Train"
			numberOfEpochs = trainNumberOfEpochs
		else:
			correct, total = 0, 0
			testLoss = 0.0
			desc="Test"
			numberOfEpochs = 1
		for epoch in tqdm(range(numberOfEpochs), desc=desc):
			trainLoss = 0.0
			for batch in tqdm(dataLoader, desc=f"Epoch {epoch + 1}", leave=False):
				if(databaseName == "MNIST"):
					x, y = batch
					x, y = x.to(device), y.to(device)
					imagesLarge = x
					batchSize, numberOfChannels, imageHeightLarge, imageWidthLarge = imagesLarge.shape
				for r in range(0, numberOfSpatialResolutions):
					if(databaseName == "MNIST"):
						if(useMultipleSpatialResolutions):
							spatialResolution = 2**r		
							spatialResolutionTransform = T.Resize((imageHeightLarge//spatialResolution, imageWidthLarge//spatialResolution))
							images = spatialResolutionTransform(imagesLarge)
						else:
							images = imagesLarge

					if(useParallelisedGeometricHashing):
						if(positionalEmbeddingTransformationOnly):
							posEmbeddingsAbsoluteGeoNormalised = ATORpt_operations.getPositionalEmbeddingsAbsolute(VITnumberOfPatches)	#or ATORnumberOfPatches
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
					if(train):
						loss = criterion(y_hat, y) / len(x)
						trainLoss += loss.detach().cpu().item()
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
					else:
						loss = criterion(y_hat, y) / len(x)
						testLoss += loss.detach().cpu().item()
						correct += pt.sum(pt.argmax(y_hat, dim=1) == y).detach().cpu().item()
						total += len(x)
			if(train):
				print(f"Epoch {epoch + 1}/{numberOfEpochs} loss: {trainLoss:.2f}")
		if(not train):
			print(f"Test loss: {testLoss:.2f}")
			print(f"Test accuracy: {correct / total * 100:.2f}%")



			
if __name__ == '__main__':
	if(useStandardVIT):
		mainStandardViT()
	else:
		mainCustomViT()
