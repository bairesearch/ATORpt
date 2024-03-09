"""ATORpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt global definitions

"""

import torch as pt
import torch.nn as nn

pt.set_printoptions(profile="full")
pt.autograd.set_detect_anomaly(True)
pt.set_default_tensor_type('torch.cuda.FloatTensor')

debugGeometricHashingParallel = True	#print geometrically transformed tensors
debugGeometricHashingParallel2 = True

useEndToEndNeuralModel = False
if(not useEndToEndNeuralModel):
	useATORPTparallel = True
	useATORCPPserial = False	#use ATOR C++ executable to generate transformed patches (normalised snapshots)

if(useEndToEndNeuralModel):
	databaseName = "MNIST"
else:
	if(useATORCPPserial):
		databaseName = "ALOI-VIEW"
	elif(useATORPTparallel):
		databaseName = "ALOI-VIEW"	#"MNIST"	#optional
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
	databaseNumberOfClasses = ALOIdatabaseNumberOfImages
	numberOfSpatialResolutions = 1
elif(databaseName == "MNIST"):
	databaseImageShape = (1, 28, 28)   #numberOfChannels, imageHeight, imageWidth
	numberOfOutputDimensions = 10
	databaseNumberOfClasses = numberOfOutputDimensions
	useMultipleSpatialResolutions = False	#feed input at multiple resolutions	#incomplete
	if(useMultipleSpatialResolutions):
		numberOfSpatialResolutions = 3
	else:
		numberOfSpatialResolutions = 1
else:
	print("unknown databaseName: = ", databaseName)
	exit()

trainNumberOfEpochs = 10

if(useEndToEndNeuralModel):
	useStandardVIT = False	#custom ViT required
	
	#initialisation (dependent vars)
	useClassificationSnapshots = False	#initialise (dependent var)
	useClassificationVIT = False	#initialise (dependent var)
	useMultKeys = False	#initialise (dependent var)
	
	batchSize = 16 #debug: 2
	ATORnumberOfPatches = 28	#inputShape[1]
	VITnumberOfPatches = 7
	usePositionalEmbeddings = True
	useParallelisedGeometricHashing = True 	#vector transformations of all image pixel coordinates in parallel
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
						useGeometricHashingHardcodedParallelisedDeformation = False	#apply multiple rotation matrices in parallel
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
	yAxis = 0	#CHECKTHIS (openCV/torchvision standard: H, W)
	xAxis = 1
else:
	useStandardVIT = True	#required
	trainVITfromScratch = True	#this is required as pretrained transformer uses positional embeddings, where as ATOR transformed patch VIT currently assumes full permutation invariance 
	if(databaseName == "ALOI-VIEW"):
		if(debugProcessSingleImage):
			batchSize = 1
		else:
			if(useATORCPPserial):
				batchSize = 1	#must process images serially (currently required for ATOR parallelised geometric hashing; assume first dimension of snapshot data in ATOR operations is patch index)
			else:
				batchSize = 2 #2, 4, 8
	else:
		batchSize = 4 #2, 4, 8
	normaliseSnapshotLength = 30
	numberOfZoomLevels = 3
	snapshotNumberOfKeypoints = 3	#tri features
	VITmaxNumberATORpatches = 900	#max number of normalised patches per image (spare patches are filled with dummy var)	#must support sqrt
	VITmaxNumberATORpolysPerZoom = VITmaxNumberATORpatches//numberOfZoomLevels	#300	#CHECKTHIS
	ATORnumberOfPatches = VITmaxNumberATORpatches
	VITnumberOfPatches = VITmaxNumberATORpatches
	VITnumberOfChannels = 3
	VITpatchSizeX = normaliseSnapshotLength
	VITimageSize = VITpatchSizeX * int(VITnumberOfPatches**0.5)
	VITpatchSize = (normaliseSnapshotLength, normaliseSnapshotLength)
	VITnumberOfPatchDimensions = VITnumberOfChannels*VITpatchSize[0]*VITpatchSize[1]
	paddingPatchTokenValue = 0	#padding patch token value 
	VITnumberOfHiddenDimensions = 512
	VITnumberOfHeads = 8
	VITnumberOfLayers = 3
	VITnumberOfClasses = databaseNumberOfClasses
	yAxis = 0	#CHECKTHIS
	xAxis = 1
	inputfolder = "/media/systemusername/large/source/ANNpython/ATORpt/ATORpt/images"	#location of ATORrules.xml, images
	numberOfGeometricDimensions = 2	#2D object data (2DOD)
	if(useATORPTparallel):
		ATORmaxNumberOfPolys = VITmaxNumberATORpatches
		keypointPadValue = -1
		ATORpatchPadding = 2	#0, 2
		ATORpatchUpscaling = 2	#0, 2
		ATORpatchSize = (normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding, normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding)	#use larger patch size to preserve information during resampling
		useGeometricHashingHardcodedParallelisedDeformation = True	#apply multiple rotation matrices in parallel
		segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_h_4b8939.pth"
		useFeatureDetectionCorners = True
		useFeatureDetectionCentroids = False	#default: True #disable for debug (speed)
		keypointDetectionRelaxedMinXYdiff = 5	#minimum difference along an X, Y axis for all 3 keypoints in a poly (used to ignore extremely elongated poly candidates)
	elif(useATORCPPserial):
		trainVITfromScratch = True	#this is required as pretrained transformer uses positional embeddings, where as ATOR transformed patch VIT currently assumes full permutation invariance 
		batchSize = 1	#process images serially
		useParallelisedGeometricHashing = False
		usePositionalEmbeddings = False
		useClassificationVIT = True
		exeFolder = "exe/" 
		ATORCexe = "ATOR.exe"
		FDCexe = "FD.exe"
		exefolder = "/media/systemusername/large/source/ANNpython/ATORpt/ATORpt/exe"	#location of ATOR.exe, FD.exe
		ATOR_DATABASE_FILESYSTEM_DEFAULT_DATABASE_NAME = "ATORfsdatabase/"	#sync with ATORdatabaseFileIO.hpp
		ATOR_DATABASE_FILESYSTEM_DEFAULT_SERVER_OR_MOUNT_NAME = "/media/systemusername/large/source/ANNpython/ATORpt/"	#sync with ATORdatabaseFileIO.hpp
		ATOR_DATABASE_CONCEPT_NAME_SUBDIRECTORY_INDEX_NUMBER_OF_LEVELS = 3 	#eg e/x/a/example
		ATOR_DATABASE_TEST_FOLDER_NAME = "test"	#sync with ATORdatabaseFileIO.hpp
		ATOR_DATABASE_TRAIN_FOLDER_NAME = "train"	#sync with ATORdatabaseFileIO.hpp
		ATOR_METHOD_2DOD_NORM_SNAPSHOT_X = 30	#sync with ATORrules.xml
		ATOR_METHOD_2DOD_NORM_SNAPSHOT_Y = 30	#sync with ATORrules.xml
		ATOR_METHOD_3DOD_NORM_SNAPSHOT_X = 40	#sync with ATORrules.xml
		ATOR_METHOD_3DOD_NORM_SNAPSHOT_Y = 40	#sync with ATORrules.xml
		ATOR_METHOD2DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS = 3	#sync with ATORrules.xml
		ATOR_METHOD3DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS = 1	#sync with ATORrules.xml
		ATOR_METHOD_POLYGON_NUMBER_OF_SIDES = 3	#sync with ATORglobalDefs.hpp	#triangle
		ATOR_DATABASE_TRAIN_TEST_FOLDER_STRUCTURE_SAME = False #sync with ATORglobalDefs.hpp
		assert (normaliseSnapshotLength == ATOR_METHOD_2DOD_NORM_SNAPSHOT_X)
		assert (numberOfZoomLevels == ATOR_METHOD2DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS)		

def printe(str):
	print(str)
	exit()
	
if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")

