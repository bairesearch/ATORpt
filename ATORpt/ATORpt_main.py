"""ATORpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and pytorch 2.2+

conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install tqdm
pip install transformers
pip install click
pip install opencv-python opencv-contrib-python
pip install kornia
pip install matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git (required for useATORPTparallel:useFeatureDetectionCentroids only)
pip install timm (required for useATORPTparallel:generate3DODfrom2DOD only)
pip install lovely-tensors

# Usage:
source activate pytorch3d
python3 ATORpt_main.py

# Description:
ATORpt main - Axis Transformation Object Recognition (ATOR)

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementations including 
receptive field feature/poly detection, parallel processed geometric hashing, end-to-end neural model. 
Classification of normalised snapshots (transformed patches) via ViT 

ATORpt contains various hardware accelerated implementations of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
	- supports classification of 2D image snapshots recreated from transformed mesh coordinates
		- perform independent, parallelised target prediction of object triangle data
- !useEndToEndNeuralModel (useStandardVIT)
	- useATORPTparallel:
		- uses parallel pytorch ATOR implementation
		- support corner/centroid features of the ATOR specification using third party libraries
		- third party feature detectors currently include, for point features: Harris/ShiTomasi/etc, centroids: segment-anything
		- supports simultaneous transformation of approx 9000 30x30px patches (ATOR 2D0D tri polys) on 12GB GPU
			- approx 10 images with 900 30x30px 2D0D tri polys per image, generated from approx 500 features per 500x500px image
			- approx 100x faster than useATORCPPserial
		- requires pytorch3d library
		- support3DOD:generate3DODfrom2DOD uses intel-isl MiDaS library (3D input object data)
	- useATORCPPserial:
		- uses ATOR C++ executable to generate transformed patches (normalised snapshots)
		- requires all ATOR C++ prerequisites 
- useEndToEndNeuralModel (!useStandardVIT):
	- contains various modules for an end-to-end neural model of ATOR
	- designed to perform transformations of all image pixel coordinates in parallel
	- architecture layout provided (implementation incomplete)
	- supports feature detection via a CNN
	- currently supports 2DOD (2D/image input object data)
	- currently uses MNIST dataset to test affine(/euclidean/projective) invariance
	- also supports useMultKeys - modify transformer to support geometric hashing operations - experimental

See ATOR specification: https://www.wipo.int/patentscope/search/en/WO2011088497

Future:
Requires upgrading to support3DOD:generate3DODfromParallax

"""

import torch as pt
import torch.nn as nn
from tqdm import tqdm

from ATORpt_globalDefs import *
import ATORpt_dataLoader
if(useStandardVIT):
	import ATORpt_vitStandard
	if(useATORCPPserial):
		import ATORpt_CPPATOR
	elif(useATORPTparallel):
		import ATORpt_PTATOR
	import torch.optim as optim
	from torchvision import transforms, datasets
	from transformers import ViTFeatureExtractor, ViTForImageClassification
else:
	import ATORpt_vitCustom
	import ATORpt_E2EATOR
	import ATORpt_E2Eoperations
	from torch.optim import Adam
	from torch.nn import CrossEntropyLoss

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

		#ATORpatchSize = ATORpt_E2Eoperations.getPatchSize(inputShape, ATORnumberOfPatches)
		patchSizeVITlocal = ATORpt_E2Eoperations.getPatchSize(inputShape, VITnumberOfPatches)

		print("inputShape = ", inputShape)
		print("patchSizeVITlocal = ", patchSizeVITlocal)
		
		if(useClassificationVIT):
			numberOfInputDimensionsVIT = ATORpt_E2Eoperations.getInputDim(inputShape, patchSizeVITlocal)
			if(useParallelisedGeometricHashing):
				numberOfHiddenDimensionsVIT = numberOfInputDimensionsVIT + numberOfGeometricDimensions2DOD   #mandatory (currently required for ATORpt_E2Eoperations.getPositionalEmbeddingsAbsolute, as no method implemented for mapping between hashing_d and numberOfHiddenDimensionsVIT)
			else:
				numberOfHiddenDimensionsVIT = 20   #arbitrary
			numberOfHeadsVIT = 1	#requirement; numberOfHiddenDimensionsVIT % numberOfHeadsVIT == 0

		modelParameters = []
		ATORmodel = None
		if(useParallelisedGeometricHashing):
			if(not positionalEmbeddingTransformationOnly):
				ATORmodel = ATORpt_E2EATOR.ATORmodelClass(inputShape, ATORnumberOfPatches)
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
			VITmodel.eval()
			correct = 0
			total = 0
			desc="Test"
			numberOfEpochs = 1

		for epoch in tqdm(range(numberOfEpochs), desc=desc):
			running_loss = 0.0
			correct = 0
			total = 0
			print("epoch = ", epoch)
			for batchIndex, batch in enumerate(tqdm(dataLoader, desc=f"Epoch {epoch + 1}", leave=False)):
				print("batchIndex = ", batchIndex)
				if(databaseName == "ALOI-VIEW"):
					imageIndices, viewIndices = batch
					print("imageIndices = ", imageIndices)
					print("viewIndices = ", viewIndices)
					labels = imageIndices
					imagePaths = ATORpt_dataLoader.getALOIVIEWImagePath(imageIndices, viewIndices)
				elif(databaseName == "MNIST"):
					x, y = batch
					x, y = x.to(device), y.to(device)
					imagesLarge = x
					_, numberOfChannels, imageHeightLarge, imageWidthLarge = imagesLarge.shape
					
				if(useATORCPPserial):
					transformedPatches = ATORpt_CPPATOR.generateATORpatches(imagePaths, train)	#normalisedsnapshots
					transformedPatches = transformedPatches.unsqueeze(0)	#add dummy batch size dimension (size 1)
				elif(useATORPTparallel):
					transformedPatches = ATORpt_PTATOR.generateATORpatches(support3DOD, imagePaths, train)	#normalisedsnapshots
				
				artificialInputImages = generateArtificialInputImages(transformedPatches)	
				print("labels = ", labels)
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
					_, numberOfChannels, imageHeightLarge, imageWidthLarge = imagesLarge.shape
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
							posEmbeddingsAbsoluteGeoNormalised = ATORpt_E2Eoperations.getPositionalEmbeddingsAbsolute(VITnumberOfPatches, xAxisViT, yAxisViT)	#or ATORnumberOfPatches
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

def generateArtificialInputImages(transformedPatches):
	#transformedPatches shape: batchSize, ATORmaxNumberOfPolys, C, H, W
	#artificialInputImages = pt.permute(0, 2, 1, 3, 4)	#shape: batchSize, C, ATORmaxNumberOfPolys, H, W
	artificialInputImages = pt.reshape(transformedPatches, (batchSize, VITnumberOfChannels, VITimageSize, VITimageSize))	#CHECKTHIS (confirm ViT artificial image creation method)
	print("artificialInputImages.shape = ", artificialInputImages.shape)
	return artificialInputImages			

			
if __name__ == '__main__':
	if(useStandardVIT):
		mainStandardViT()
	else:
		mainCustomViT()
