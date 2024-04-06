"""ATORpt_vitStandard.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt vit standard

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *
from transformers import ViTModel
from transformers import ViTConfig

if(trainVITfromScratch):
	class ViTForImageClassificationClass(nn.Module):
		def __init__(self):
			super(ViTForImageClassificationClass, self).__init__()
			image_size = VITimageSize
			patch_size = VITpatchSize
			num_classes = VITnumberOfClasses
			hidden_dim = VITnumberOfHiddenDimensions
			num_heads = VITnumberOfHeads
			num_layers = VITnumberOfLayers
			patch_dim = VITnumberOfPatchDimensions
			patch_channels = VITnumberOfChannels

			self.patch_size = patch_size
			self.patch_embedding = nn.Conv2d(patch_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
			encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
			self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
			self.classification_head = nn.Linear(hidden_dim, num_classes)  # No positional embedding

		def forward(self, x):
			patches = self.patch_embedding(x)
			print("patches.shape = ", patches.shape)
			batch_size, _, h, w = patches.shape
			patches = patches.view(batch_size, -1, h * w).permute(0, 2, 1)
			print("patches.shape = ", patches.shape)
			encoded_features = self.transformer_encoder(patches)
			print("encoded_features.shape = ", encoded_features.shape)
			logits = self.classification_head(encoded_features[:, 0])  # Use only the first token
			return logits
else:
	config = ViTConfig(
		image_size = VITpatchSize,
		channels = VITnumberOfChannels,
		num_classes = VITnumberOfClasses
	)
	class ViTClass(nn.Module):
		def __init__(self):
			super(ViTClass, self).__init__()
			self.vit = ViTModel(config)

		def forward(self, x):
			return self.vit(x)

def getVITimageSize(transformedPatches):	#not used (for clarity only)
	batchSize, numberOfPatches, numberOfChannels, patchSize, _ = transformedPatches.shape
	imageSize = patchSize * int(numberOfPatches**0.5)
	return imageSize


