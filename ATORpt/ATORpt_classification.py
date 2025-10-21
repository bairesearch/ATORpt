"""ATORpt_classification.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
None

# Installation:
see ATORpt_main.py

# Description:
ATORpt classification

"""

import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from ATORpt_globalDefs import *

snapshotMajorityTopK = 1  # number of top classes to keep when tallying snapshot votes per image	#min(3, VITnumberOfClasses)

_vit_expected_image_size = None
_vit_image_mean = None
_vit_image_std = None
_vit_rescale_factor = None

def configure_vit_preprocessor(image_size, image_mean=None, image_std=None, rescale_factor=None):
	global _vit_expected_image_size, _vit_image_mean, _vit_image_std, _vit_rescale_factor
	if image_size is not None:
		if isinstance(image_size, (tuple, list)):
			if len(image_size) == 2:
				_vit_expected_image_size = tuple(int(s) for s in image_size)
			else:
				size_value = int(image_size[0])
				_vit_expected_image_size = (size_value, size_value)
		else:
			size_value = int(image_size)
			_vit_expected_image_size = (size_value, size_value)
	else:
		_vit_expected_image_size = None
	if image_mean is not None:
		_vit_image_mean = pt.tensor(image_mean).view(1, -1, 1, 1)
	else:
		_vit_image_mean = None
	if image_std is not None:
		_vit_image_std = pt.tensor(image_std).view(1, -1, 1, 1)
	else:
		_vit_image_std = None
	_vit_rescale_factor = rescale_factor

def _apply_vit_preprocessing(tensor):
	if _vit_expected_image_size is not None:
		target_height, target_width = _vit_expected_image_size
		if tensor.shape[-2] != target_height or tensor.shape[-1] != target_width:
			tensor = F.interpolate(tensor, size=(target_height, target_width), mode="bilinear", align_corners=False)
	if _vit_rescale_factor is not None:
		tensor = tensor * _vit_rescale_factor
	# fall back to scaling 0-255 inputs if no rescale factor provided
	elif tensor.max() > 1.0:
		tensor = tensor / 255.0
	if _vit_image_mean is not None and _vit_image_std is not None:
		mean = _vit_image_mean.to(tensor.device, dtype=tensor.dtype)
		std = _vit_image_std.to(tensor.device, dtype=tensor.dtype)
		tensor = (tensor - mean) / std
	return tensor

def flattenNormalisedSnapshotBatch(transformedPatches):
	# transformedPatches shape: batchSize, ATORmaxNumberOfPolys, C, H, W
	batchSizeDynamic, numSnapshots, channels, height, width = transformedPatches.shape
	neuralModelInputs = transformedPatches.reshape(batchSizeDynamic * numSnapshots, channels, height, width)
	if(debugVerbose):
		print("flattenNormalisedSnapshotBatch: neuralModelInputs.shape = ", neuralModelInputs.shape)
	return neuralModelInputs, batchSizeDynamic, numSnapshots

def extractVITLogits(neuralModelOutputs):
	return neuralModelOutputs.logits if hasattr(neuralModelOutputs, "logits") else neuralModelOutputs

def forwardClassificationNeuralModelOnSnapshots(neuralModel, transformedPatches):
	neuralModelInputs, batchSizeDynamic, numSnapshots = flattenNormalisedSnapshotBatch(transformedPatches)
	model_device = next(neuralModel.parameters()).device
	neuralModelInputs = neuralModelInputs.to(model_device, dtype=pt.float32)
	if(hasattr(neuralModel, "config") and neuralModel.config.model_type == "vit"):
		neuralModelInputs = _apply_vit_preprocessing(neuralModelInputs)
	neuralModelOutputs = neuralModel(neuralModelInputs)
	logits = extractVITLogits(neuralModelOutputs)
	logitsPerSnapshot = logits.view(batchSizeDynamic, numSnapshots, -1)
	logitsPerImage = logitsPerSnapshot.mean(dim=1)
	snapshotPredictions = logitsPerSnapshot.argmax(dim=2)
	classCounts = F.one_hot(snapshotPredictions, num_classes=logitsPerSnapshot.shape[-1]).sum(dim=1).to(pt.float32)
	topkK = snapshotMajorityTopK	#min(snapshotMajorityTopK, classCounts.shape[-1])
	if(topkK < 1):
		raise ValueError("snapshotMajorityTopK must be >= 1 for majority voting")
	topkCounts, topkIndices = classCounts.topk(topkK, dim=1)
	majorityPredictions = topkIndices[:, 0]
	if(debugVerbose):
		print("forwardClassificationNeuralModelOnSnapshots: logitsPerSnapshot.shape = ", logitsPerSnapshot.shape)
		print("forwardClassificationNeuralModelOnSnapshots: classCounts.shape = ", classCounts.shape)
	return logitsPerImage, logitsPerSnapshot, majorityPredictions, topkIndices, topkCounts
	
