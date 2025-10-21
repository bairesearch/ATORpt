"""ATORpt_classificationVITstandard.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt classification vit standard

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
				patch_channels = VITnumberOfChannels

				if isinstance(image_size, int):
					img_height = img_width = image_size
				else:
					img_height, img_width = image_size
				num_patches = (img_height // patch_size[0]) * (img_width // patch_size[1])

				self.patch_size = patch_size
				self.patch_embedding = nn.Conv2d(patch_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
				self.class_token = nn.Parameter(pt.zeros(1, 1, hidden_dim))
				self.pos_embedding = nn.Parameter(pt.zeros(1, num_patches + 1, hidden_dim))
				self.dropout = nn.Dropout(0.1)
				encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
				self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
				self.encoder_norm = nn.LayerNorm(hidden_dim)
				self.classification_head = nn.Linear(hidden_dim, num_classes)
				nn.init.trunc_normal_(self.class_token, std=0.02)
				nn.init.trunc_normal_(self.pos_embedding, std=0.02)

			def forward(self, x):
				patches = self.patch_embedding(x)
				batch_size = patches.shape[0]
				patches = patches.flatten(2).transpose(1, 2)
				cls_tokens = self.class_token.expand(batch_size, -1, -1)
				tokens = pt.cat((cls_tokens, patches), dim=1)
				tokens = tokens + self.pos_embedding[:, :tokens.size(1)]
				tokens = self.dropout(tokens)
				encoded_features = self.transformer_encoder(tokens)
				encoded_features = self.encoder_norm(encoded_features)
				logits = self.classification_head(encoded_features[:, 0])  # Use cls token
				if(debugVerbose):
					print("x.shape = ", x.shape)
					print("patches.shape = ", patches.shape)
					print("encoded_features.shape = ", encoded_features.shape)
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
