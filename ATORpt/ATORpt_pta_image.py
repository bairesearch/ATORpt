"""ATORpt_pta_image.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt pta image functions (pytorch equivalent to tensorflow addons tfa image functions)

"""

#equivalent pytorch functions for tensorflow_addons.image.translate, tensorflow_addons.image.rotate, tensorflow_addons.image.transform,
import kornia.augmentation as Kaugmentation
import kornia.geometry.transform as Ktransform

import torch as pt
import torch.nn as nn

def scale(image_tensor, scaleFactor, fillValue=0):
	#scaleMatrix = [scaleFactor, 0.0, 0.0, 0.0, scaleFactor, 0.0, 0.0, 0.0]
	#transformedImage = tfa.image.transform(image_tensor, scaleMatrix)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	transform = Ktransform.Scale(scaleFactor, scaleFactor, padding_mode='constant', fill_value=fillValue)
	transformed_image = transform(image_tensor)
	return transformed_image

def translate(image_tensor, centerCoordinatesList, fillValue=0):
	#RFfilterTransformed = tfa_image.translate(image_tensor, centerCoordinatesList, fill_value=fillValue) 	#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	transform = K.RandomAffine(degrees=0.0, translate=centerCoordinatesList, padding_mode='constant', fill_value=fillValue) 
	transformed_image = transform(image_tensor) 
	return transformed_image
	
def rotate(image_tensor, centerCoordinatesList, fillValue=0):
	#RFfilterTransformed = tfa_image.rotate(image_tensor, angleRadians, fill_value=fillValue)	# https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate
	transformed_image = TF.rotate(image_tensor, angle)
	return transformed_image
	
#def transform(image_tensor, transformation_matrix, fillValue=0):
#	transformed_image = K.warp_perspective(image_tensor, transformation_matrix, (image_tensor.shape[2], image_tensor.shape[3]), padding_mode='constant', fill_value=fillValue) 
