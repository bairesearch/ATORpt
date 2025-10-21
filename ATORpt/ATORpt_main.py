"""ATORpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and pytorch

---
ATORpt_globalDefs.useATORPTparallel=True: 
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
python -m pip install "numpy==1.24.4"
pip install tqdm
python -m pip install "transformers==4.37.0"
pip install click
python -m pip install --upgrade --force-reinstall "opencv-python==4.7.0.72" "opencv-contrib-python==4.7.0.72"
pip install kornia
pip install matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git (required for useATORPTparallel:useFeatureDetectionCentroids and ATORpt_RFmainSA)
pip install timm (required for useATORPTparallel:generate3DODfrom2DOD only)
pip install lovely-tensors
	python -m pip install --force-reinstall --no-deps "opencv-python==4.7.0.72" "opencv-contrib-python==4.7.0.72"
	python -m pip install --force-reinstall --no-deps git+https://github.com/facebookresearch/segment-anything.git
	python -m pip install --force-reinstall --no-deps lovely-tensors
	python -m pip install --upgrade --force-reinstall "numpy==1.24.4"

---
ATORpt_globalDefs.useATORRFparallel=True: 
conda create -n sam2 python=3.12
conda activate sam2
pip install torch torchvision torchaudio
pip install tqdm
pip install transformers
pip install click
pip install opencv-python opencv-contrib-python
pip install kornia
pip install matplotlib
pip install git+https://github.com/facebookresearch/sam2.git
pip install timm (required for useATORPTparallel:generate3DODfrom2DOD only)
pip install lovely-tensors
pip install accelerate

# Usage:
source activate pytorch3d
python3 ATORpt_main.py

source activate sam2
python3 ATORpt_main.py

# Description:
ATORpt main - Axis Transformation Object Recognition (ATOR)

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementations including 
receptive field feature/poly detection, parallel processed geometric hashing, end-to-end neural model. 
Classification of normalised snapshots (transformed patches) via ViT 

ATORpt contains various hardware accelerated implementations of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a neural model (eg vit) - experimental
	- supports classification of 2D image snapshots recreated from transformed mesh coordinates
		- perform independent, parallelised target prediction of object triangle data
- useClassificationNeuralModel (if classificationModelName=="VIT": useStandardVIT)
	- useATORRFparallel
		- uses ATOR RF to generate normalised snapshots
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
import torch.nn.functional as F
from tqdm import tqdm

from ATORpt_globalDefs import *
import ATORpt_dataLoader
if(useClassificationNeuralModel):
	if(useATORRFparallel):
		import ATORpt_RFmainSA
	elif(useATORPTparallel):
		import ATORpt_PTATOR
	elif(useATORCPPserial):
		import ATORpt_CPPATOR
	import torch.optim as optim
	from torchvision import transforms, datasets
	import ATORpt_classification
	classificationModelNameUpper = classificationModelName.upper()
	if(classificationModelNameUpper=="VIT"):
		import ATORpt_classificationVITstandard
		from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoConfig
	else:
		if(classificationModelNameUpper=="MLP"):
			import ATORpt_classificationMLP
		elif(classificationModelNameUpper=="CNN"):
			import ATORpt_classificationCNN
		else:
			raise ValueError(f"Unsupported classificationModelName: {classificationModelName}")
else:
	import ATORpt_E2EATOR
	import ATORpt_E2Eoperations
	from torch.optim import Adam
	from torch.nn import CrossEntropyLoss
	if(classificationModelName=="VIT"):
		import ATORpt_classificationVITcustom
	else:
		printe("useEndToEndNeuralModel requires custom VIT")
		
device = pt.device('cuda')


if(useClassificationNeuralModel):
	def mainClassificationNeuralModel():
		#print("mainClassificationNeuralModel")
		trainDataLoader, testDataLoader = ATORpt_dataLoader.createDataloader()

		if(classificationModelNameUpper=="VIT"):
			if(trainVITfromScratch):
				neuralModel = ATORpt_classificationVITstandard.ViTForImageClassificationClass()
			else:
				feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
				config = AutoConfig.from_pretrained('google/vit-base-patch16-224-in21k')
				config.num_labels = databaseNumberOfClasses
				# match the classifier head to the dataset; pretrained head weights are discarded
				config.id2label = {i: str(i) for i in range(databaseNumberOfClasses)}
				config.label2id = {label: idx for idx, label in config.id2label.items()}
				ATORpt_classification.configure_vit_preprocessor(
					image_size=config.image_size,
					image_mean=getattr(feature_extractor, "image_mean", None),
					image_std=getattr(feature_extractor, "image_std", None),
					rescale_factor=getattr(feature_extractor, "rescale_factor", None),
				)
				neuralModel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', config=config, ignore_mismatched_sizes=True)
		else:
			if(classificationModelNameUpper=="MLP"):
				neuralModel = ATORpt_classificationMLP.SnapshotMLPClassifier()
			elif(classificationModelNameUpper=="CNN"):
				neuralModel = ATORpt_classificationCNN.SnapshotCNNClassifier()
			else:
				raise ValueError(f"Unsupported classificationModelName: {classificationModelName}")

		neuralModel = neuralModel.to(device)

		trainOrTestClassificationNeuralModel(True, trainDataLoader, neuralModel)
		trainOrTestClassificationNeuralModel(False, testDataLoader, neuralModel)
else:
	def mainEndToEndNeuralModel():
		assert classificationModelName=="VIT", "useEndToEndNeuralModel requires custom VIT"
		
		#template: https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py

		trainDataLoader, testDataLoader = ATORpt_dataLoader.createDataloader()
		inputShape = databaseImageShape

		#ATORpatchSize = ATORpt_E2Eoperations.getPatchSize(inputShape, ATORnumberOfPatches)
		patchSizeVITlocal = ATORpt_E2Eoperations.getPatchSize(inputShape, VITnumberOfPatches)

		print("inputShape = ", inputShape)
		print("patchSizeVITlocal = ", patchSizeVITlocal)

		if(useE2EclassificationVIT):
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
		if(useE2EclassificationSnapshots):
			neuralModel = None
		if(useE2EclassificationVIT):
			neuralModel = ATORpt_classificationVITcustom.ViTClass(inputShape, VITnumberOfPatches, numberOfHiddenDimensionsVIT, numberOfHeadsVIT, numberOfOutputDimensions)
			modelParameters = modelParameters + list(neuralModel.parameters()) 

		trainOrTestEndToEndNeuralModel(True, trainDataLoader, ATORmodel, neuralModel, modelParameters)
		trainOrTestEndToEndNeuralModel(False, testDataLoader, ATORmodel, neuralModel, modelParameters)
		
if(useClassificationNeuralModel):
	def trainOrTestClassificationNeuralModel(train, dataLoader, neuralModel):
		print("trainOrTestClassificationNeuralModel: train = ", str(train))
		criterion = nn.CrossEntropyLoss()
		if(train):
			if(classificationModelNameUpper=="VIT"):
				if(trainVITfromScratch):
					learningRate = 1e-4
					for param in neuralModel.parameters():
						param.requires_grad = True
				else:
					learningRate = 1e-4
					for param in neuralModel.parameters():
						param.requires_grad = False
					if(hasattr(neuralModel, "classifier")):
						headParameters = neuralModel.classifier.parameters()
					elif(hasattr(neuralModel, "classification_head")):
						headParameters = neuralModel.classification_head.parameters()
					else:
						raise AttributeError("Unexpected ViT classification head attribute name")
					for param in headParameters:
						param.requires_grad = True
			else:
				learningRate = 1e-3
				for param in neuralModel.parameters():
					param.requires_grad = True
			neuralModel.train()
			desc="Train"
			numberOfEpochs = trainNumberOfEpochs
			trainableParameters = [param for param in neuralModel.parameters() if param.requires_grad]
			if(len(trainableParameters) == 0):
				trainableParameters = list(neuralModel.parameters())
			optimizer = optim.Adam(trainableParameters, lr=learningRate)
		else:
			neuralModel.eval()
			correct = 0
			total = 0
			desc="Test"
			numberOfEpochs = 1

		#for epoch in tqdm(range(numberOfEpochs), desc=desc):
		for epoch in range(numberOfEpochs):
			running_loss = 0.0
			correct = 0
			total = 0
			#print("epoch = ", epoch)
			progressBar = tqdm(dataLoader, desc=f"Epoch {epoch + 1} ({desc})", leave=False)
			for batchIndex, batch in enumerate(progressBar):
				if(debugVerbose):
					print("batchIndex = ", batchIndex)
				if(databaseName == "ALOI-VIEW"):
					imageIndices, viewIndices = batch
					if(debugVerbose):
						print("imageIndices = ", imageIndices)
						print("viewIndices = ", viewIndices)
					labels = imageIndices
					imagePaths = ATORpt_dataLoader.getALOIVIEWImagePath(imageIndices, viewIndices)
					if(debugVerbose):
						print("imagePaths = ", imagePaths)
				else:
					x, y = batch
					x, y = x.to(device), y.to(device)
					imagesLarge = x
					labels = y
					_, numberOfChannels, imageHeightLarge, imageWidthLarge = imagesLarge.shape
					imagePaths = [imageTensor.detach().cpu() for imageTensor in imagesLarge]
					
				if(useATORRFparallel):
					transformedPatches = ATORpt_RFmainSA.generateATORpatches(support3DOD, imagePaths, train)	#normalisedsnapshots
				elif(useATORPTparallel):
					transformedPatches = ATORpt_PTATOR.generateATORpatches(support3DOD, imagePaths, train)	#normalisedsnapshots
				elif(useATORCPPserial):
					transformedPatches = ATORpt_CPPATOR.generateATORpatches(imagePaths, train)	#normalisedsnapshots
					transformedPatches = transformedPatches.unsqueeze(0)	#add dummy batch size dimension (size 1)
				
				logitsPerImage, logitsPerSnapshot, predicted, batchTopkIndices, batchTopkCounts = ATORpt_classification.forwardClassificationNeuralModelOnSnapshots(neuralModel, transformedPatches)
				if(isinstance(labels, pt.Tensor)):
					labels = labels.to(logitsPerImage.device, non_blocking=True).long()
				else:
					labels = pt.as_tensor(labels, device=logitsPerImage.device, dtype=pt.long)
				loss_image = criterion(logitsPerImage, labels)
				numSnapshots = logitsPerSnapshot.shape[1]
				labels_expanded = labels.unsqueeze(1).expand(-1, numSnapshots).reshape(-1)
				logits_snapshot_flat = logitsPerSnapshot.reshape(-1, logitsPerSnapshot.shape[-1])
				loss_snapshots = criterion(logits_snapshot_flat, labels_expanded)
				if(processSnapshotLossSeparately):
					loss = loss_snapshots
					current_loss_value = loss_snapshots.item()
				else:
					loss = loss_image
					current_loss_value = loss_image.item()
				if(debugMajoritySnapshotClassificationVoting):
					print("labels = ", labels)
					print("batchTopkIndices = ", batchTopkIndices)
					print("batchTopkCounts = ", batchTopkCounts)
				if(isinstance(predicted, pt.Tensor)):
					predicted = predicted.to(labels.device).detach()
				else:
					predicted = pt.as_tensor(predicted, device=labels.device)
				if(train):
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				if(predicted.dtype != labels.dtype):
					predicted = predicted.to(dtype=labels.dtype)
				batch_correct = (predicted == labels).sum().item()
				batch_total = labels.size(0)
				batch_accuracy = (batch_correct / batch_total) if batch_total > 0 else 0.0
				running_loss += current_loss_value
				total += batch_total
				correct += batch_correct
				postfix = {
					"loss": f"{current_loss_value:.4f}",
					"acc": f"{batch_accuracy * 100:.2f}%"
				}
				if(processSnapshotLossSeparately):
					postfix["loss_img"] = f"{loss_image.item():.4f}"
				else:
					postfix["loss_snap"] = f"{loss_snapshots.item():.4f}"
				progressBar.set_postfix(**postfix)
			progressBar.close()
			if(train):
				train_loss = running_loss / len(dataLoader)
				train_acc = correct / total
				print(f'Epoch [{epoch+1}/{numberOfEpochs}], Loss: {train_loss:.4f}, Accuracy: {100*train_acc:.2f}%')
		if(not train):
			test_acc = correct / total
			print(f'Test Accuracy: {100*test_acc:.2f}%')
else:
	def trainOrTestEndToEndNeuralModel(train, dataLoader, ATORmodel, neuralModel, modelParameters):
		print("trainOrTestEndToEndNeuralModel: train = ", str(train))
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
		#for epoch in tqdm(range(numberOfEpochs), desc=desc):
		for epoch in range(numberOfEpochs):
			trainLoss = 0.0
			for batch in tqdm(dataLoader, desc=f"Epoch {epoch + 1}", leave=False):
				x, y = batch
				x, y = x.to(device), y.to(device)
				imagesLarge = x
				_, numberOfChannels, imageHeightLarge, imageWidthLarge = imagesLarge.shape
				for r in range(0, numberOfSpatialResolutions):
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

					if(useE2EclassificationSnapshots):
						#recreate a 2D image from the transformed mesh before performing object recognition
							#perform independent/parallised target prediction of object triangle data
						print("warning: useE2EclassificationSnapshots is incomplete")
					elif(useE2EclassificationVIT):
						y_hat = neuralModel(images, posEmbeddingsAbsoluteGeoNormalised)
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
	if(useClassificationNeuralModel):
		mainClassificationNeuralModel()
	else:
		mainEndToEndNeuralModel()
