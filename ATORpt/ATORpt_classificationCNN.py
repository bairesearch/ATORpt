"""ATORpt_classificationCNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
None

# Installation:
see ATORpt_main.py

# Description:
ATORpt classification CNN

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *

class SnapshotCNNClassifier(nn.Module):
	def __init__(self, inputChannels=VITnumberOfChannels, numClasses=databaseNumberOfClasses, dropout=0.3):
		super(SnapshotCNNClassifier, self).__init__()
		self.featureExtractor = nn.Sequential(
			nn.Conv2d(inputChannels, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1))
		)

		self.embedding = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout)
		)

		self.classification_head = nn.Linear(256, numClasses)

	def forward(self, x):
		x = self.featureExtractor(x)
		x = self.embedding(x)
		logits = self.classification_head(x)
		if(debugVerbose):
			print("SnapshotCNNClassifier forward: logits.shape = ", logits.shape)
		return logits
