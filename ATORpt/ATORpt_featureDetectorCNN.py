"""ATORpt_featureDetectorCNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt feature detector CNN
s
"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *
import ATORpt_operations

class FeatureDetectorCNNClass(nn.Module):
	def __init__(self, numberOfPatches):
		super(FeatureDetectorCNNClass, self).__init__()

		self.numberOfPatches = numberOfPatches
		self.numChannels = 16
		self.featureDetector = nn.Sequential(
				nn.Conv2d(in_channels=1, out_channels=self.numChannels, kernel_size=5, stride=1, padding='same'),
				nn.ReLU(),
				nn.Conv2d(in_channels=self.numChannels, out_channels=1, kernel_size=3, stride=1, padding='same'),
				nn.ReLU()
		)

	def forward(self, images):
	
		images = ATORpt_operations.normaliseInputs0to1(images, dim=None)	#normalise pixels across all dimensions
	
		featureMap = self.featureDetector(images)

		#convert 2D to linear;
		featureMap = ATORpt_operations.createLinearPatches(featureMap, self.numberOfPatches)		

		return featureMap

