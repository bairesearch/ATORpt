"""ATORpt_pta_image.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt pta image functions (pytorch equivalent to tensorflow addons tfa image functions)

"""

#equivalent pytorch functions for tensorflow_addons.image.translate, tensorflow_addons.image.rotate, tensorflow_addons.image.transform,
#import kornia.augmentation as Kaugmentation
#import kornia.geometry.transform as Ktransform
#import kornia as K
import torch as pt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import math
pta_image_rotate_doesNotSupportCUDA = True
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def scale(image_tensor, scaleFactor, fillValue=0):
	#scaleMatrix = [scaleFactor, 0.0, 0.0, 0.0, scaleFactor, 0.0, 0.0, 0.0]
	#transformedImage = tfa.image.transform(image_tensor, scaleMatrix)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	image_tensor = image_tensor.unsqueeze(0)
	scale_matrix = pt.tensor([[scaleFactor, 0.0, 0.0], [0.0, scaleFactor, 0.0]], device=device)
	scale_matrix = scale_matrix.unsqueeze(0)
	grid = F.affine_grid(scale_matrix, image_tensor.size())
	transformed_image = F.grid_sample(image_tensor, grid)
	transformed_image = transformed_image.squeeze(0)
	return transformed_image

def translate(image_tensor, centerCoordinatesList, fillValue=0):
	#RFfilterTransformed = tfa_image.translate(image_tensor, centerCoordinatesList, fill_value=fillValue) 	#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	image_tensor = image_tensor.unsqueeze(0)
	translation_matrix = pt.tensor([[1, 0, centerCoordinatesList[0]], [0, 1, centerCoordinatesList[1]]], dtype=pt.float, device=device)
	translation_matrix = translation_matrix.unsqueeze(0)
	grid = F.affine_grid(translation_matrix, image_tensor.size())
	transformed_image = F.grid_sample(image_tensor, grid, padding_mode='border', align_corners=True)
	transformed_image = transformed_image.squeeze(0)
	return transformed_image
	
def rotate(image_tensor, angleRadians, fillValue=0):
	angleDegrees = angleRadians * (180.0 / math.pi)
	pil_image = TF.to_pil_image(image_tensor)
	transformed_image = TF.rotate(pil_image, angleDegrees)
	transformed_image = TF.to_tensor(transformed_image)
	if(pta_image_rotate_doesNotSupportCUDA):
		transformed_image = transformed_image.to(device)
	'''
	angleRadians = pt.tensor(angleRadians, dtype=pt.float)
	image_tensor = image_tensor.unsqueeze(0)
	rotation_matrix = pt.tensor([[pt.cos(angleRadians), -pt.sin(angleRadians), 0], [pt.sin(angleRadians), pt.cos(angleRadians), 0]])
	rotation_matrix = rotation_matrix.unsqueeze(0)
	grid = F.affine_grid(rotation_matrix, image_tensor.size())
	rotated_image = F.grid_sample(image_tensor, grid, padding_mode='border', align_corners=True)
	transformed_image = rotated_image.squeeze(0)
	'''
	return transformed_image

#def transform(image_tensor, transformation_matrix, fillValue=0):
#	transformed_image = K.warp_perspective(image_tensor, transformation_matrix, (image_tensor.shape[2], image_tensor.shape[3]), padding_mode='constant', fill_value=fillValue) 
