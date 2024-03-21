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
import math

pt.set_printoptions(profile="full")
pt.autograd.set_detect_anomaly(True)
pt.set_default_tensor_type('torch.cuda.FloatTensor')

#debug vars:
debugProcessSingleImage = False
debugSnapshotRenderFullImage = False	#draw original image to debug the renderer
debugSnapshotRenderFinal = False	#draw final transformed snapshots to debug the renderer
debugSnapshotRender = False	#draw intermediary transformed snapshots to debug the renderer
debugGeometricHashingParallelFinal = False
debugGeometricHashingParallel = False	#print intermediary transformed keypoints to debug the geometric hashing
debugGeometricHashingParallelLargeMarker = True
debugSnapshotRenderCroppedImage = False		#draw cropped images (preprocessing of untransformed snapshots) to debug the image coordinates generation for geometric hashing/rendering
debugFeatureDetection = False	#print features on original images
debugPolyIndex = 0	#poly index used for intermediary transformed snapshots debugging 

#ATOR implementation select:
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
	databaseRoot = "/media/rich/datasets/ALOI-VIEW/" 
	databaseImageShape = (3, 768, 576)   #numberOfChannels, imageHeight, imageWidth
	numberOfOutputDimensions = 1000	
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
				batchSize = 2 #2, 4, 8	#depend on GPU ram (VITmaxNumberATORpatches, ATORpatchPadding)
	else:
		batchSize = 4 #2, 4, 8
	normaliseSnapshotLength = 30
	numberOfZoomLevels = 3
	snapshotNumberOfKeypoints = 3	#tri features
	if(debugGeometricHashingParallel or debugSnapshotRender or debugGeometricHashingParallelFinal or debugSnapshotRenderFinal):
		VITmaxNumberATORpatches = 30
	else: 
		VITmaxNumberATORpatches = 900	#max number of normalised patches per image (spare patches are filled with dummy var)	#lower number required for debug (CUDA memory)
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
	inputfolder = "/media/rich/large/source/ANNpython/ATORpt/ATORpt/images"	#location of ATORrules.xml, images
	numberOfGeometricDimensions = 2	#2D object data (2DOD)
	if(useATORPTparallel):		
		fullRotationalInvariance = False	#optional	#requires 3x GPU ram #create keypoint sets for every poly orientation (x3) - else assume input image is roughly upright; only perform 1 geometric hashing transformation (not geometricHashingNumKeypoints transformations, based on every possible permutation of keypoints)
		if(fullRotationalInvariance):
			assert (VITmaxNumberATORpatches%snapshotNumberOfKeypoints == 0)	#ensure VITmaxNumberATORpatches is divisible by snapshotNumberOfKeypoints
		keypointAindex = 0
		keypointBindex = 1
		keypointCindex = 2
		ATORmaxNumImages = 10	#max 10 on 12GB GPU
		ATORmaxNumberATORpatchesAllImages = VITmaxNumberATORpatches*ATORmaxNumImages	#max 9000 on 12GB GPU
		ATORmaxNumberOfPolys = VITmaxNumberATORpatches	#max number of normalised patches per image
		keypointPadValue = -1
		meshPadValue = -1
		ATORpatchPadding = 2	#1, 2
		ATORpatchUpscaling = 1	#1, 2
		ATORpatchSizeIntermediary = (normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding, normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding)	#use larger patch size to preserve information during resampling
		useGeometricHashingHardcodedParallelisedDeformation = True	#apply multiple rotation matrices in parallel
		segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_h_4b8939.pth"
		useFeatureDetectionCorners = True
		useFeatureDetectionCentroids = True	#default: True #disable for debug (speed)
		keypointDetectionCriteria = True
		if(keypointDetectionCriteria):
			keypointDetectionMaxSimilarity = 1.0	#in pixels
			keypointDetectionMinXYdiff = 5	#minimum difference along X, Y axis in pixels for all 3 keypoints in a poly (used to ignore extremely elongated poly candidates)
			keypointDetectionMinApexYDiff = 2	#minimum difference of Y axis apex of object triangle
			#keypointDetectionMinBaseXDiff = 2	#minimum difference along an X axis for base of object triangle
			keypointDetectionMaxColinearity = 1.0 #0.3	#as a proportion of X/Y distance off line
		ATORmaxNumberOfNearestFeaturesToSamplePolyKeypoints = 3	#must be >= 2
		snapshotRenderer = "pytorch3D"
		normalisedObjectTriangleBaseLength = 1
		normalisedObjectTriangleHeight = 1	#1: use equal base length and height for square snapshot generation, math.sqrt(3)/2: use equilateral triangle
		if(debugSnapshotRender):
			#renderViewportSize = (normalisedObjectTriangleBaseLength*2, normalisedObjectTriangleHeight*2)	#increase size of snapshot area to see if image coordinates align with object triangle
			renderViewportSize = (normalisedObjectTriangleBaseLength, normalisedObjectTriangleHeight)
		else:
			renderViewportSize = (normalisedObjectTriangleBaseLength, normalisedObjectTriangleHeight)
			#normaliseSnapshotLength*ATORpatchPadding
		renderImageSize = normaliseSnapshotLength
		if(ATORpatchPadding == 1):
			applyObjectTriangleMask = True	#mask out transformed image coordinates outside of object triangle
		else:
			applyObjectTriangleMask = False
		if(snapshotRenderer == "pytorch3D"):
			snapshotRenderTris = True	#else quads	#snapshots must be rendered using artificial Tri polygons (generated from pixel quads)
			snapshotRenderExpectColorsDefinedForVerticesNotFaces = True
			if(snapshotRenderExpectColorsDefinedForVerticesNotFaces):
				snapshotRenderExpectColorsDefinedForVerticesNotFacesPadVal = 0
			renderInvertedYaxisToDisplayOriginalImagesUpright = False	 #orient camera to face up wrt original images (required as opencv/TF image y coordinates are defined from top to bottom)
			if(renderInvertedYaxisToDisplayOriginalImagesUpright):
				snapshotRenderCameraRotationZaxis = 180
			else:
				snapshotRenderCameraRotationZaxis = 0
			snapshotRenderCameraRotationYaxis = 0	#orient camera to face towards the mesh
			snapshotRenderCameraRotationXaxis = 0
			if(snapshotRenderCameraRotationYaxis == 180):
				snapshotRenderCameraZnear = 0.0
				snapshotRenderCameraZfar = -100.0
				snapshotRenderZdimVal = -10.0
			else:
				snapshotRenderCameraZnear = 0.1
				snapshotRenderCameraZfar = 100.0
				snapshotRenderZdimVal = 10.0
	elif(useATORCPPserial):
		VITmaxNumberATORpolysPerZoom = VITmaxNumberATORpatches//numberOfZoomLevels	#300	#CHECKTHIS
		trainVITfromScratch = True	#this is required as pretrained transformer uses positional embeddings, where as ATOR transformed patch VIT currently assumes full permutation invariance 
		batchSize = 1	#process images serially
		useParallelisedGeometricHashing = False
		usePositionalEmbeddings = False
		useClassificationVIT = True
		exeFolder = "exe/" 
		ATORCexe = "ATOR.exe"
		FDCexe = "FD.exe"
		exefolder = "/media/rich/large/source/ANNpython/ATORpt/ATORpt/exe"	#location of ATOR.exe, FD.exe
		ATOR_DATABASE_FILESYSTEM_DEFAULT_DATABASE_NAME = "ATORfsdatabase/"	#sync with ATORdatabaseFileIO.hpp
		ATOR_DATABASE_FILESYSTEM_DEFAULT_SERVER_OR_MOUNT_NAME = "/media/rich/large/source/ANNpython/ATORpt/"	#sync with ATORdatabaseFileIO.hpp
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

xAxisATORmodel = 0	#ATOR model assumes x,y coordinates
yAxisATORmodel = 1
xAxisGeometricHashing = xAxisATORmodel	#geometric hashing assumes x,y coordinates (used by renderer also)
yAxisGeometricHashing = yAxisATORmodel
xAxisFeatureMap = xAxisATORmodel	#ATOR feature map assumes x,y coordinates
yAxisFeatureMap = yAxisATORmodel
xAxisViT = 1	#ViT assumes y,x patch coordinates (standard opencv/TF image coordinates convention also)
xAxisViT = 0
xAxisImages = 1	#opencv/torchvision(TF/PIL)/etc assume y,x coordinates
yAxisImages = 0
#matplotlib imshow assumes y,x,c; opencv->tensor assumes c,y,x, TFPIL->tensor assumes c,y,x)
	
def printe(str):
	print(str)
	exit()
	
if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")

