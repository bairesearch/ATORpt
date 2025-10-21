"""ATORpt_classificationMLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
None

# Installation:
see ATORpt_main.py

# Description:
ATORpt classification MLP

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *

class SnapshotMLPClassifier(nn.Module):
	def __init__(self, inputChannels=VITnumberOfChannels, patchSize=VITpatchSize, numClasses=databaseNumberOfClasses, hiddenDims=None, dropout=0.3):
		super(SnapshotMLPClassifier, self).__init__()
		height, width = patchSize
		self.inputDim = inputChannels * height * width
		if(hiddenDims is None):
			hiddenDims = [1024, 512]

		layers = []
		prevDim = self.inputDim
		for hiddenDim in hiddenDims:
			layers.append(nn.Linear(prevDim, hiddenDim))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.Dropout(dropout))
			prevDim = hiddenDim

		if(len(layers) > 0):
			self.featureExtractor = nn.Sequential(*layers)
		else:
			self.featureExtractor = nn.Identity()

		self.classification_head = nn.Linear(prevDim, numClasses)

	def forward(self, x):
		x = x.reshape(x.size(0), -1)
		x = self.featureExtractor(x)
		logits = self.classification_head(x)
		if(debugVerbose):
			print("SnapshotMLPClassifier forward: x.shape = ", x.shape)
			print("SnapshotMLPClassifier forward: logits.shape = ", logits.shape)
		return logits
