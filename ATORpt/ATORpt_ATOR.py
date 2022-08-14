"""ATORpt_ATOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt geometric hashing

"""

import torch as pt
import torch.nn as nn

import torchvision.transforms as T

from ATORpt_globalDefs import *
import ATORpt_geometricHashing
import ATORpt_featureDetectorCNN
import ATORpt_AMANN
import ATORpt_operations
#from torchdim import dims

if(debugGeometricHashingHardcoded):
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm


class ATORmodelClass(nn.Module):
	def __init__(self, inputShape, numberOfPatches):
		super(ATORmodelClass, self).__init__()
				
		self.inputShape = inputShape
		numberOfChannels, imageHeight, imageWidth = inputShape
		self.numberOfPatches = numberOfPatches

		self.patchSize = ATORpt_operations.getPatchSize(inputShape, numberOfPatches)

		self.numberOfInputDimensions = ATORpt_operations.getInputDim(inputShape, self.patchSize)

		self.featureDetectorPatchSize = ATORpt_operations.getPatchSize(inputShape, imageWidth)
		self.featureDetectorInputDim = ATORpt_operations.getInputDim(inputShape, self.featureDetectorPatchSize)
		self.featureDetectorNumPatches = imageWidth

		self.sequenceLength = ATORpt_operations.getInputLayerNumTokens(numberOfPatches)
		
		if(useGeometricHashingCNNfeatureDetector):
			self.featureDetectorCNN = ATORpt_featureDetectorCNN.FeatureDetectorCNNClass(self.featureDetectorNumPatches)
		else:
			pass
			#use custom feature detector (e.g. Heitger et al.)

		self.geometricHashing = ATORpt_geometricHashing.GeometricHashingClass(self.featureDetectorInputDim, self.featureDetectorNumPatches)

	def forward(self, images):
		
		batchSize, numberOfChannels, imageHeight, imageWidth  = images.shape
					
		tokens = ATORpt_operations.createLinearPatches(images, self.numberOfPatches)
		posEmbeddings = ATORpt_operations.getPositionalEmbeddingsAbsolute(self.numberOfPatches)
		
		#ATORpt_operations.printPixelMap(posEmbeddings, tokens)
		
		featureDetectorInput = images

		if(useGeometricHashingCNNfeatureDetector):
			featureMap = self.featureDetectorCNN(featureDetectorInput)	#featureMap has been linearised
		else:
			pass
			#featureMap = self.featureDetectorMSA(featureDetectorInput)

		posEmbeddingsAbsoluteGeoNormalised = self.geometricHashing(posEmbeddings, tokens, featureMap)

		return posEmbeddingsAbsoluteGeoNormalised


