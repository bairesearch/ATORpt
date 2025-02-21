"""ATORpt_PTglobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT global definitions

"""

from ATORpt_globalDefs import * 


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
	ATORpatchPadding2DOD = 2	#1, 2
	if(support3DOD):
		ATOR3DODobjectTriangleMaintinAspectRatio = True	#3DOD object triangle is not deformed to an equilateral triangle; it maintains its original aspect ratio	 #ATOR3DODobjectTriangleMaintinAspectRatio is orig ATOR 3DOD implementation	#only currently available implementation
		if(ATOR3DODobjectTriangleMaintinAspectRatio):
			ATOR3DODrenderViewportSizeExpand = 2 	#used to ensure all object triangle data are captured in viewport	#will depend on both keypointDetectionMaxColinearity and ATORpatchPadding
		else:
			ATOR3DODrenderViewportSizeExpand = 1
		ATORpatchPadding3DOD = ATORpatchPadding2DOD*ATOR3DODrenderViewportSizeExpand
	ATORpatchUpscaling = 1	#1, 2
	ATORpatchSizeIntermediary2DOD = (normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding2DOD, normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding2DOD)	#use larger patch size to preserve information during resampling
	ATORpatchCropPaddingValue = 0	#must match https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html #If image size is smaller than output size along any edge, image is padded with 0 and then cropped.
	if(support3DOD):
		ATORpatchSizeIntermediary3DOD = (normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding3DOD, normaliseSnapshotLength*ATORpatchUpscaling*ATORpatchPadding3DOD)	#use larger patch size to preserve information during resampling
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
	if(support3DOD):
		ATOR3DODsetKeypointDepthMinimum = True	#set the keypoint detection depth/Z to their closest value 
		ATOR3DODgeoHashingAlignObjectTriangleBaseVertically = True	#align object triangle base with y axis 	#currently required (else must reconfigure eye, up, at)
		if(ATOR3DODgeoHashingScale):
			renderViewportSize3DOD = (normalisedObjectTriangleBaseLength*ATOR3DODrenderViewportSizeExpand, normalisedObjectTriangleHeight*ATOR3DODrenderViewportSizeExpand)
		else:
			renderViewportSize3DOD = (600, 600)	#max image size	#CHECKTHIS
	if(debugSnapshotRender):
		#renderViewportSize2DOD = (normalisedObjectTriangleBaseLength*2, normalisedObjectTriangleHeight*2)	#increase size of snapshot area to see if image coordinates align with object triangle
		renderViewportSize2DOD = (normalisedObjectTriangleBaseLength, normalisedObjectTriangleHeight)
	else:
		renderViewportSize2DOD = (normalisedObjectTriangleBaseLength, normalisedObjectTriangleHeight)
		#normaliseSnapshotLength*ATORpatchPadding
	renderImageSize = normaliseSnapshotLength
	if(ATORpatchPadding2DOD == 1):
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
			if(support3DOD):
				snapshotRenderCameraZnear = 100.0
				snapshotRenderCameraZfar = -100.0
				snapshotRenderCameraZworkaround = True	#workaround required as FoVOrthographicCameras does not appear to render both -Z and +Z coordinates, irrespective of how snapshotRenderCameraZnear/snapshotRenderCameraZfar are set
			else:
				snapshotRenderCameraZnear = 0.0
				snapshotRenderCameraZfar = -100.0
			snapshotRenderZdimVal = -10.0
		else:
			if(support3DOD):
				snapshotRenderCameraZnear = -100.0
				snapshotRenderCameraZfar = 100.0
				snapshotRenderCameraZworkaround = True	#workaround required as FoVOrthographicCameras does not appear to render both -Z and +Z coordinates, irrespective of how snapshotRenderCameraZnear/snapshotRenderCameraZfar are set
			else:
				snapshotRenderCameraZnear = 0.1
				snapshotRenderCameraZfar = 100.0
			snapshotRenderZdimVal = 10.0

elif(useATORCPPserial):
	VITmaxNumberATORpolysPerZoom = VITmaxNumberATORpatches//numberOfZoomLevels	#300	#CHECKTHIS
	batchSize = 1	#process images serially
	useParallelisedGeometricHashing = False
	usePositionalEmbeddings = False
	useClassificationVIT = True
	exeFolder = "exe/" 
	ATORCexe = "ATOR.exe"
	FDCexe = "FD.exe"
	exefolder = "/media/" + userName + "/large/source/ANNpython/ATORpt/ATORpt/exe"	#location of ATOR.exe, FD.exe
	ATOR_DATABASE_FILESYSTEM_DEFAULT_DATABASE_NAME = "ATORfsdatabase/"	#sync with ATORdatabaseFileIO.hpp
	ATOR_DATABASE_FILESYSTEM_DEFAULT_SERVER_OR_MOUNT_NAME = "/media/" + userName + "/large/source/ANNpython/ATORpt/"	#sync with ATORdatabaseFileIO.hpp
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

