"""ATORpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt global definitions

"""

import torch as pt
import torch.nn as nn

pt.set_printoptions(profile="full")
pt.autograd.set_detect_anomaly(True)
pt.set_default_tensor_type('torch.cuda.FloatTensor')


debugGeometricHashingHardcoded = False	#print geometrically transformed tensors

useMultipleSpatialResolutions = False	#feed input at multiple resolutions	#incomplete
if(useMultipleSpatialResolutions):
	numberOfSpatialResolutions = 3
else:
	numberOfSpatialResolutions = 1

#initialisation (dependent vars)
useClassificationSnapshots = False	#initialise (dependent var)
useClassificationVIT = False	#initialise (dependent var)
useMultKeys = False	#initialise (dependent var)

useParallelisedGeometricHashing = True 	#vector transformations of all image pixel coordinates in parallel
if(useParallelisedGeometricHashing):
	useClassificationSnapshots = True	#optional	#perform classification of 2D image snapshots recreated from transformed mesh coordinates - standard (see C++ ATOR implementation) #incomplete
	useClassificationVIT = False	#optional	#perform classification of transformed coordinates with a vision transformer (vit) - experimental

	useGeometricHashingProbabilisticKeypoints = False   #for backprop   #else use topk  #optional
	useGeometricHashingCNNfeatureDetector = True   #mandatory
	useGeometricHashingPixels = True	#mandatory
	if(useGeometricHashingProbabilisticKeypoints):
		useGeometricHashingAMANN = True #mandatory	#experimental
		useGeometricHashingProbabilisticKeypointsSoftMax = False
		useGeometricHashingProbabilisticKeypointsNonlinearity = True
		if(useGeometricHashingProbabilisticKeypointsNonlinearity):
			if(useGeometricHashingProbabilisticKeypointsSoftMax):
				useGeometricHashingProbabilisticKeypointsNonlinearityOffset = 0.022	#input Z is normalised between 0 and 1 #calibrate
			else:
				useGeometricHashingProbabilisticKeypointsNonlinearityOffset = 1.0  #calibrate
		useGeometricHashingProbabilisticKeypointsZero = True	#zero all keypoints below attention threshold  #CHECKTHIS: affects backprop 
		if(useGeometricHashingAMANN):
			useGeometricHashingNormaliseOutput = True	#normalise geometric hashed positional embeddings output from 0 to 1
			useGeometricHashingReduceInputMagnitude = False
		else:
			print("useGeometricHashingHardcoded not supported for useGeometricHashingProbabilisticKeypoints")
	else:
		useGeometricHashingAMANN = False #mandatory  #experimental	#else use hardcoded geohashing function
		if(useGeometricHashingAMANN):
			useGeometricHashingNormaliseOutput = True
			useGeometricHashingReduceInputMagnitude = False #reduce average magnitude of positional embedding input
		else:
			useGeometricHashingHardcoded = True
			if(useGeometricHashingHardcoded):
				useGeometricHashingHardcodedParallelisedRotation = False	#requires implementation (apply multiple rotation matrices in parallel)
	useGeometricHashingKeypointNormalisation = True
	numberOfGeometricDimensions = 2	#2D object data (2DOD)
else:
	useMultKeys = True   #experimental (modify transformer to support geometric hashing operations)
	if(useMultKeys):
		useClassificationVIT = True
	
activationMaxVal = 10.0
multiplicativeEmulationFunctionOffsetVal = 1.0	#add/subtract
multiplicativeEmulationFunctionPreMinVal = 1e-9
multiplicativeEmulationFunctionPreMaxVal = 1e+9	#or activationMaxVal (effective)
multiplicativeEmulationFunctionPostMaxVal = 20.0

yAxis = 0	#CHECKTHIS
xAxis = 1


