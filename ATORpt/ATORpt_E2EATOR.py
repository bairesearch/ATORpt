"""ATORpt_E2EATOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E ATOR

"""

import torch as pt
import torch.nn as nn

import torchvision.transforms as T

from ATORpt_globalDefs import *
import ATORpt_E2EgeometricHashing
import ATORpt_E2Efeatures
import ATORpt_E2EAMANN
import ATORpt_E2Eoperations
#from torchdim import dims

if(debugGeometricHashingParallel):
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm


class ATORmodelClass(nn.Module):
	def __init__(self, inputShape, numberOfPatches):
		super(ATORmodelClass, self).__init__()
				
		self.inputShape = inputShape
		numberOfChannels, imageHeight, imageWidth = inputShape
		self.numberOfPatches = numberOfPatches	#image width/height (ATOR uses a unique "patch" for each pixel)

		self.patchSize = ATORpt_E2Eoperations.getPatchSize(inputShape, numberOfPatches)	#1, 1

		self.numberOfInputDimensions = ATORpt_E2Eoperations.getInputDim(inputShape, self.patchSize)

		self.featureDetectorPatchSize = ATORpt_E2Eoperations.getPatchSize(inputShape, imageWidth)
		self.featureDetectorInputDim = ATORpt_E2Eoperations.getInputDim(inputShape, self.featureDetectorPatchSize)
		self.featureDetectorNumPatches = imageWidth

		self.sequenceLength = ATORpt_E2Eoperations.getInputLayerNumTokens(numberOfPatches)
		
		if(useGeometricHashingCNNfeatureDetector):
			self.featureDetectorCNN = ATORpt_E2Efeatures.FeatureDetectorCNNClass(self.featureDetectorNumPatches)
		else:
			pass
			#use custom feature detector (e.g. Heitger et al.)

		self.geometricHashing = ATORpt_E2EgeometricHashing.GeometricHashingClass(self.featureDetectorInputDim, self.featureDetectorNumPatches)

	def forward(self, images):
		
		images = images.permute(0, 1, 3, 2)	#place image into C,W,H format (as ATOR model uses xAxisATORmodel=0,yAxisATORmodel=1 convention)
		batchSize, numberOfChannels, imageWidth, imageHeight = images.shape
					
		tokens = ATORpt_E2Eoperations.createLinearPatches(images, self.numberOfPatches, True)
		posEmbeddings = ATORpt_E2Eoperations.getPositionalEmbeddingsAbsolute(self.numberOfPatches, xAxisATORmodel, yAxisATORmodel)
		
		#ATORpt_E2Eoperations.printPixelMap(posEmbeddings, tokens)
		
		featureDetectorInput = images

		if(useGeometricHashingCNNfeatureDetector):
			featureMap = self.featureDetectorCNN(featureDetectorInput)	#featureMap has been linearised
		else:
			pass
			#featureMap = self.featureDetectorMSA(featureDetectorInput)

		posEmbeddingsAbsoluteGeoNormalised = self.geometricHashing(images, posEmbeddings, tokens, featureMap)

		return posEmbeddingsAbsoluteGeoNormalised


