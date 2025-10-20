"""ATORpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

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
import os


#****** algorithm selection parameters ***********

useEndToEndNeuralModel = False
if(useEndToEndNeuralModel):
	useStandardVIT = False	#custom ViT required
else:
	useStandardVIT = True	#required
	trainVITfromScratch = True	#this is required as pretrained transformer uses positional embeddings, where as ATOR transformed patch VIT currently assumes full permutation invariance 
	useATORRFparallel = True
	useATORPTparallel = False
	useATORCPPserial = False	#use ATOR C++ executable to generate transformed patches (normalised snapshots)


#****** debug parameters ***********

#debug vars:
debugMajoritySnapshotClassificationVoting = True
debugSingleZoomLevel = False
debugVITbasic = True
debugVITlow = False
debugVITmoderate = False
debugProcessSingleImage = False
debugVerbose = False
debugATOR = False
if(debugATOR):
	debug3DODgeneration = False
	if(debug3DODgeneration):
		debugVITmaxNumberATORpatches = 4	#required for !ATOR3DODgeoHashingScale
	else:
		debugVITmaxNumberATORpatches = 36	#60	#90	#30

#from ATORpt_PTglobalDefs
#if(debugATOR):
debugSnapshotRenderFullImage = False	#draw original image to debug the renderer
debugSnapshotRenderFinal = False	#draw final transformed snapshots to debug the renderer
debugSnapshotRender = False	#draw intermediary transformed snapshots to debug the renderer
debugGeometricHashingParallelFinal = False	#verify that final transformation results in normalised snapshot (*)
debugGeometricHashingParallel = False	#print intermediary transformed keypoints to debug the geometric hashing
debugSnapshotRenderCroppedImage = False		#draw cropped images (preprocessing of untransformed snapshots) to debug the image coordinates generation for geometric hashing/rendering
debugFeatureDetection = False	#print features on original images
debugPolyIndex = 0	#poly index used for intermediary transformed snapshots debugging 
debugGeometricHashingParallelLargeMarker = True


#****** 3DOD parameters ***********
	
#ATOR implementation select:
support3DOD = False	#incomplete
if(support3DOD):
	generate3DODfrom2DOD = True	#generate 3DOD (3d object data) from image before executing ATOR
	#generate3DODfromParallax = False	#not yet coded	#parallax resolvable depth (PRD-ATOR)
	ATOR3DODgeoHashingScale = False	#!ATOR3DODgeoHashingScale is orig ATOR 3DOD implementation (will create full mesh snapshots rather than object triangle snapshots)
	
	
#****** database parameters ***********

userName = 'systemusername'	#default: systemusername
if(os.path.isdir('user')):
	from user.user_globalDefs import *
	
if(useEndToEndNeuralModel):
	databaseName = "MNIST"
else:
	if(useATORRFparallel):
		databaseName = "ALOI-VIEW"
	if(useATORPTparallel):
		databaseName = "ALOI-VIEW"	#"MNIST"	#optional
	elif(useATORCPPserial):
		databaseName = "ALOI-VIEW"
	
if(databaseName == "ALOI-VIEW"):
	databaseRoot = "/media/" + userName + "/datasets/ALOI-VIEW/" 
	databaseImageShape = (3, 768, 576)   #numberOfChannels, imageHeight, imageWidth
	ALOIdatabaseImageStartIndex = 1
	if(debugVITbasic):
		numberOfOutputDimensions = 10	#1000
		ALOIdatabaseNumberOfImages = 10	#1000
		ALOIdatabaseNumberOfViews = 4	#72
		ALOIdatabaseNumberOfIlluminationDirections = 4	#24
		ALOIdatabaseNumberOfIlluminationColours = 4	#12
		ALOIdatabaseNumberOfViewsTrain = 3	#64
		ALOIdatabaseNumberOfViewsTest = 1	#8
		databaseNumberOfClasses = ALOIdatabaseNumberOfImages
		numberOfSpatialResolutions = 1	
	elif(debugVITlow):
		numberOfOutputDimensions = 10	#1000
		ALOIdatabaseNumberOfImages = 10	#1000
		ALOIdatabaseNumberOfViews = 4	#72
		ALOIdatabaseNumberOfIlluminationDirections = 24	#24
		ALOIdatabaseNumberOfIlluminationColours = 12	#12
		ALOIdatabaseNumberOfViewsTrain = 3	#64
		ALOIdatabaseNumberOfViewsTest = 1	#8
		databaseNumberOfClasses = ALOIdatabaseNumberOfImages
		numberOfSpatialResolutions = 1		
	elif(debugVITmoderate):
		numberOfOutputDimensions = 10	
		ALOIdatabaseNumberOfImages = 10
		ALOIdatabaseNumberOfViews = 72
		ALOIdatabaseNumberOfIlluminationDirections = 24
		ALOIdatabaseNumberOfIlluminationColours = 12
		ALOIdatabaseNumberOfViewsTrain = 64
		ALOIdatabaseNumberOfViewsTest = 8
		databaseNumberOfClasses = ALOIdatabaseNumberOfImages
		numberOfSpatialResolutions = 1
		'''
		numberOfOutputDimensions = 100	#1000
		ALOIdatabaseNumberOfImages = 100	#1000
		ALOIdatabaseNumberOfViews = 4	#72
		ALOIdatabaseNumberOfIlluminationDirections = 24	#24
		ALOIdatabaseNumberOfIlluminationColours = 12	#12
		ALOIdatabaseNumberOfViewsTrain = 3	#64
		ALOIdatabaseNumberOfViewsTest = 1	#8
		databaseNumberOfClasses = ALOIdatabaseNumberOfImages
		numberOfSpatialResolutions = 1
		'''
	else:
		numberOfOutputDimensions = 1000	
		ALOIdatabaseNumberOfImages = 1000
		ALOIdatabaseNumberOfViews = 72
		ALOIdatabaseNumberOfIlluminationDirections = 24
		ALOIdatabaseNumberOfIlluminationColours = 12
		ALOIdatabaseNumberOfViewsTrain = 64
		ALOIdatabaseNumberOfViewsTest = 8
		databaseNumberOfClasses = ALOIdatabaseNumberOfImages
		numberOfSpatialResolutions = 1
	if(debugProcessSingleImage):
		debugProcessSingleImageIndexTrain = 868	#Object.nr:.868 - nutrilon nora box	#Object.nr:.525 - Paper box	#common ATOR C implementation samples for high point/corner feature detection
		debugProcessSingleViewIndexTrain = 0
		debugProcessSingleImageIndexTest = 868	#Object.nr:.868 - nutrilon nora box	#Object.nr:.525 - Paper box	#common ATOR C implementation samples for high point/corner feature detection
		debugProcessSingleViewIndexTest = 66
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

databaseTrainShuffle = True

trainNumberOfEpochs = 10	#10	#1


#****** ViT parameters ***********

if(databaseName == "ALOI-VIEW"):
	if(debugProcessSingleImage):
		batchSize = 1
	else:
		if(useATORRFparallel):
			batchSize = 8 #2, 4, 8	#depend on GPU ram (VITmaxNumberATORpatches, ATORpatchPadding)
		elif(useATORPTparallel):
			batchSize = 2 #2, 4, 8	#depend on GPU ram (VITmaxNumberATORpatches, ATORpatchPadding)
		elif(useATORCPPserial):
			batchSize = 1	#must process images serially (currently required for ATOR parallelised geometric hashing; assume first dimension of snapshot data in ATOR operations is patch index)
else:
	batchSize = 4 #2, 4, 8
if(useATORRFparallel):
	normaliseSnapshotLength = 60	#there are less snapshots per image with ATOR RF, so snapshots can be larger
else:
	normaliseSnapshotLength = 30
if(debugSingleZoomLevel):
	numberOfZoomLevels = 1
else:
	numberOfZoomLevels = 3
snapshotNumberOfKeypoints = 3	#tri features	#numberCoordinatesInSnapshot
if(debugATOR):	#debugGeometricHashingParallel or debugSnapshotRender or debugGeometricHashingParallelFinal or debugSnapshotRenderFinal):
	VITmaxNumberATORpatches = debugVITmaxNumberATORpatches
else: 
	if(useATORRFparallel):
		VITmaxNumberATORpatches = 64
	else:
		if(support3DOD):
			if(ATOR3DODgeoHashingScale):
				VITmaxNumberATORpatches = 400	#ATOR3DODrenderViewportSizeExpand requires more GPU ram
			else:
				VITmaxNumberATORpatches = 16	#!ATOR3DODgeoHashingScale currently requires very high GPU ram (~full image snapshots)
		else:
			VITmaxNumberATORpatches = 900	#max number of normalised patches per image (spare patches are filled with dummy var)	#lower number required for debug (CUDA memory)
#print("VITmaxNumberATORpatches**0.5 = ", VITmaxNumberATORpatches**0.5)
assert ((VITmaxNumberATORpatches**0.5)%1 == 0)	#ensure sqrt(VITmaxNumberATORpatches) is a whole number
VITnumberOfPatches = VITmaxNumberATORpatches
VITnumberOfChannels = 3
VITpatchSizeX = normaliseSnapshotLength
VITimageSize = VITpatchSizeX * int(VITnumberOfPatches**0.5)
VITpatchSize = (normaliseSnapshotLength, normaliseSnapshotLength)
VITnumberOfPatchDimensions = VITnumberOfChannels*VITpatchSize[0]*VITpatchSize[1]
paddingPatchTokenValue = 0	#padding patch token value 
VITnumberOfHiddenDimensions = 512	#default: 512
VITnumberOfHeads = 8
VITnumberOfLayers = 3
VITnumberOfClasses = databaseNumberOfClasses
inputfolder = "/media/" + userName + "/large/source/ANNpython/ATORpt/ATORpt/images"	#location of ATORrules.xml, images
numberOfGeometricDimensions2DOD = 2	#2D object data (2DOD)
numberOfGeometricDimensions3DOD = 3	#3D object data (3DOD)

print("batchSize = ", batchSize)
print("VITpatchSize = ", VITpatchSize)
print("VITimageSize = ", VITimageSize)
print("VITnumberOfPatches = ", VITnumberOfPatches)


#****** axis parameters ***********

xAxisATORmodel = 0	#ATOR model assumes x,y coordinates
yAxisATORmodel = 1
zAxisATORmodel = 2
xAxisGeometricHashing = xAxisATORmodel	#geometric hashing assumes x,y coordinates (used by renderer also)
yAxisGeometricHashing = yAxisATORmodel
zAxisGeometricHashing = zAxisATORmodel
xAxisFeatureMap = xAxisATORmodel	#ATOR feature map assumes x,y coordinates
yAxisFeatureMap = yAxisATORmodel
xAxisViT = 1	#ViT assumes y,x patch coordinates (standard opencv/TF image coordinates convention also)
xAxisViT = 0
xAxisImages = 1	#opencv/torchvision(TF/PIL)/etc assume y,x coordinates
yAxisImages = 0
#matplotlib imshow assumes y,x,c; opencv->tensor assumes c,y,x, TFPIL->tensor assumes c,y,x)


#****** segment-anything parameters ***********

segmentAnythingViTHSAMname = "vit_h"	#default
segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_h_4b8939.pth"	#default
#segmentAnythingViTHSAMname = "vit_l"	#lower GPU memory
#segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_l_0b3195.pth"
#segmentAnythingViTHSAMname = "vit_b"	#lower GPU memory
#segmentAnythingViTHSAMpathName = "../segmentAnythingViTHSAM/sam_vit_b_01ec64.pth"


#****** pytorch parameters ***********

def printe(str):
	print(str)
	exit()
	
if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")
devicePreprocessing = pt.device("cuda")	#orig: pt.device("cpu")  #image preprocessing transformation operations are currently performed on GPU/CPU

useLovelyTensors = True
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pt.set_printoptions(profile="full")
pt.autograd.set_detect_anomaly(True)
pt.set_default_tensor_type('torch.cuda.FloatTensor')
