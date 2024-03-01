"""ATORpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

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

#initialisation (dependent vars)
useClassificationSnapshots = False	#initialise (dependent var)
useClassificationVIT = False	#initialise (dependent var)
useMultKeys = False	#initialise (dependent var)

databaseName = "ALOI-VIEW"
#databaseName = "MNIST"
if(databaseName == "ALOI-VIEW"):
	databaseRoot = "/media/systemusername/datasets/ALOI-VIEW/" 
	databaseImageShape = (3, 768, 576)   #numberOfChannels, imageHeight, imageWidth
	numberOfOutputDimensions = 1000	
	debugProcessSingleImage = True
	if(debugProcessSingleImage):
		debugProcessSingleImageIndexTrain = 868	#Object.nr:.868 - nutrilon nora box	#Object.nr:.525 - Paper box	#common ATOR C implementation samples for high point/corner feature detection
		debugProcessSingleViewIndexTrain = 0
		debugProcessSingleImageIndexTest = 868	#Object.nr:.868 - nutrilon nora box	#Object.nr:.525 - Paper box	#common ATOR C implementation samples for high point/corner feature detection
		debugProcessSingleViewIndexTest = 66
	ALOIdatabaseNumberOfImages = 1000
	ALOIdatabaseNumberOfViews = 72
	ALOIdatabaseNumberOfIlluminationDirections = 24
	ALOIdatabaseNumberOfIlluminationColours = 12
	ALOIdatabaseNumberOfViewsTrain = 64
	ALOIdatabaseNumberOfViewsTest = 8
	numberOfSpatialResolutions = 1
elif(databaseName == "MNIST"):
	databaseImageShape = (1, 28, 28)   #numberOfChannels, imageHeight, imageWidth
	numberOfOutputDimensions = 10
	useMultipleSpatialResolutions = False	#feed input at multiple resolutions	#incomplete
	if(useMultipleSpatialResolutions):
		numberOfSpatialResolutions = 3
	else:
		numberOfSpatialResolutions = 1
else:
	print("unknown databaseName: = ", databaseName)
	exit()

trainNumberOfEpochs = 10

useATORCserialGeometricHashing = True	#use ATOR C++ executable to generate transformed patches (normalised snapshots)
if(useATORCserialGeometricHashing):
	useStandardVIT = True	#currently required
	trainVITfromScratch = True	#this is required as pretrained transformer uses positional embeddings, where as ATOR transformed patch VIT currently assumes full permutation invariance 
	exeFolder = "exe/" 
	ATORCexe = "ATOR.exe"
	FDCexe = "FD.exe"
	batchSize = 1	#process images serially
	useParallelisedGeometricHashing = False
	usePositionalEmbeddings = False
	useClassificationVIT = True
	inputfolder = "/media/systemusername/large/source/ANNpython/ATORpt/ATORpt/images"	#location of ATORrules.xml, images
	exefolder = "/media/systemusername/large/source/ANNpython/ATORpt/ATORpt/exe"	#location of ATOR.exe, FD.exe
	ATOR_DATABASE_FILESYSTEM_DEFAULT_DATABASE_NAME = "ATORfsdatabase/"	#sync with ATORdatabaseFileIO.hpp
	ATOR_DATABASE_FILESYSTEM_DEFAULT_SERVER_OR_MOUNT_NAME = "/media/systemusername/large/source/ANNpython/ATORpt/"	#sync with ATORdatabaseFileIO.hpp
	ATOR_DATABASE_CONCEPT_NAME_SUBDIRECTORY_INDEX_NUMBER_OF_LEVELS = 3 	#eg e/x/a/example
	ATOR_DATABASE_TEST_FOLDER_NAME = "test"	#sync with ATORdatabaseFileIO.hpp
	ATOR_DATABASE_TRAIN_FOLDER_NAME = "train"	#sync with ATORdatabaseFileIO.hpp
	ATOR_METHOD_2DOD_NORM_SNAPSHOT = 30	#sync with ATORrules.xml
	ATOR_METHOD_2DOD_NORM_SNAPSHOT = 30	#sync with ATORrules.xml
	ATOR_METHOD_3DOD_NORM_SNAPSHOT = 40	#sync with ATORrules.xml
	ATOR_METHOD_3DOD_NORM_SNAPSHOT = 40	#sync with ATORrules.xml
	ATOR_METHOD2DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS = 3	#sync with ATORrules.xml
	ATOR_METHOD3DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS = 1	#sync with ATORrules.xml
	ATOR_METHOD_POLYGON_NUMBER_OF_SIDES = 3	#sync with ATORglobalDefs.hpp	#triangle
	ATOR_DATABASE_TRAIN_TEST_FOLDER_STRUCTURE_SAME = False #sync with ATORglobalDefs.hpp
	VITmaxNumberATORpatches = 900	#max number of normalised patches per image (spare patches are filled with dummy var)	#must support sqrt
	VITmaxNumberATORpolys = 300	#CHECKTHIS
	ATORnumberOfPatches = VITmaxNumberATORpatches
	VITnumberOfPatches = VITmaxNumberATORpatches
	VITnumberOfChannels = 3
	ATORpatchSize = (ATOR_METHOD_2DOD_NORM_SNAPSHOT, ATOR_METHOD_2DOD_NORM_SNAPSHOT)
	if(useStandardVIT):
		VITpatchSizeX = ATOR_METHOD_2DOD_NORM_SNAPSHOT
		VITimageSize = VITpatchSizeX * int(VITnumberOfPatches**0.5)
		VITpatchSize = (ATOR_METHOD_2DOD_NORM_SNAPSHOT, ATOR_METHOD_2DOD_NORM_SNAPSHOT)
	else:
		VITpatchSize = (ATOR_METHOD_2DOD_NORM_SNAPSHOT, ATOR_METHOD_2DOD_NORM_SNAPSHOT)
	VITnumberOfPatchDimensions = VITnumberOfChannels*VITpatchSize[0]*VITpatchSize[1]
	paddingPatchTokenValue = 0	#padding patch token value 
	VITnumberOfHiddenDimensions = 512
	VITnumberOfHeads = 8
	VITnumberOfLayers = 3
	VITnumberOfClasses = ALOIdatabaseNumberOfImages
else:
	numberOfEpochs
	useStandardVIT = False
	batchSize = 16 #debug: 2
	ATORnumberOfPatches = 28	#inputShape[1]
	VITnumberOfPatches = 7
	useParallelisedGeometricHashing = True 	#vector transformations of all image pixel coordinates in parallel
	usePositionalEmbeddings = True
	if(useParallelisedGeometricHashing):
		positionalEmbeddingTransformationOnly = False	#perform vit positional embedding transformation (leave feature tokens unmodified), do not apply dedicated ATOR/geometric hashing (ie if False apply dedicated ATOR/geometric hashing)
		if(positionalEmbeddingTransformationOnly):
			useClassificationSnapshots = False	#optional	#perform classification of 2D image snapshots recreated from transformed mesh coordinates - standard (see C++ ATOR implementation) #incomplete
			useClassificationVIT = True	#optional	#perform classification of transformed coordinates with a vision transformer (vit) - experimental
		else:
			#orig: useClassificationSnapshots=True, useClassificationVIT=False
			useClassificationSnapshots = False	#optional	#perform classification of 2D image snapshots recreated from transformed mesh coordinates - standard (see C++ ATOR implementation) #incomplete
			useClassificationVIT = True	#optional	#perform classification of transformed coordinates with a vision transformer (vit) - experimental

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

debugGeometricHashingHardcoded = True	#print geometrically transformed tensors

def printe(str):
	print(str)
	exit()

