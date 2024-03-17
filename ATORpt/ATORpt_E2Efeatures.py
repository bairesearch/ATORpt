"""ATORpt_E2Efeatures.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E features

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *
import ATORpt_E2Eoperations

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
	
		images = ATORpt_E2Eoperations.normaliseInputs0to1(images, dim=None)	#normalise pixels across all dimensions
	
		featureMap = self.featureDetector(images)

		#convert 2D to linear;
		featureMap = ATORpt_E2Eoperations.createLinearPatches(featureMap, self.numberOfPatches)		

		return featureMap

