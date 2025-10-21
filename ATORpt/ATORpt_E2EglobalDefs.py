"""ATORpt_E2EglobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E global definitions

"""

from ATORpt_globalDefs import * 


#initialisation (dependent vars)
useE2EclassificationSnapshots = False	#initialise (dependent var)
useE2EclassificationVIT = False	#initialise (dependent var)
useMultKeys = False	#initialise (dependent var)

batchSize = 16 #debug: 2
ATORnumberOfPatches = 28	#inputShape[1]
VITnumberOfPatches = 7
usePositionalEmbeddings = True
useParallelisedGeometricHashing = True 	#vector transformations of all image pixel coordinates in parallel
if(useParallelisedGeometricHashing):
	positionalEmbeddingTransformationOnly = False	#perform vit positional embedding transformation (leave feature tokens unmodified), do not apply dedicated ATOR/geometric hashing (ie if False apply dedicated ATOR/geometric hashing)
	if(positionalEmbeddingTransformationOnly):
		useE2EclassificationSnapshots = False	#optional	#perform classification of 2D image snapshots recreated from transformed mesh coordinates - standard (see C++ ATOR implementation) #incomplete
		useE2EclassificationVIT = True	#optional	#perform classification of transformed coordinates with a vision transformer (vit) - experimental
	else:
		#orig: useE2EclassificationSnapshots=True, useE2EclassificationVIT=False
		useE2EclassificationSnapshots = False	#optional	#perform classification of 2D image snapshots recreated from transformed mesh coordinates - standard (see C++ ATOR implementation) #incomplete
		useE2EclassificationVIT = True	#optional	#perform classification of transformed coordinates with a vision transformer (vit) - experimental

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
					useGeometricHashingHardcodedParallelisedDeformation = False	#apply multiple rotation matrices in parallel
		useGeometricHashingKeypointNormalisation = True
	numberOfGeometricDimensions2DOD = 2	#2D object data (2DOD)
else:
	useMultKeys = True   #experimental (modify transformer to support geometric hashing operations)
	if(useMultKeys):
		useE2EclassificationVIT = True
activationMaxVal = 10.0
multiplicativeEmulationFunctionOffsetVal = 1.0	#add/subtract
multiplicativeEmulationFunctionPreMinVal = 1e-9
multiplicativeEmulationFunctionPreMaxVal = 1e+9	#or activationMaxVal (effective)
multiplicativeEmulationFunctionPostMaxVal = 20.0
