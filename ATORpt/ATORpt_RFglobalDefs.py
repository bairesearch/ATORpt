"""ATORpt_RFglobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Description:
ATORpt RF global definitions

"""
import numpy as np
import sys

#hardware acceleration 
RFuseParallelProcessedCNN = False	#added 03 Mar 2024	#parallel processing implementation using CNN
RFuseParallelProcessedCNNRFchannelsImplementation = 0
if(RFuseParallelProcessedCNN):
	ensureMinimumImageSizeGreaterThanRFsize = True	#required
	RFuseParallelProcessedCNNRFasChannels = True
	RFuseParallelProcessedCNNRFchannelsImplementation = 2	#apply separate CNN per RFfiltersTensor	#currently required by ATORpt_RFgenerateApply:applyRFfilters
		#RFuseParallelProcessedCNNRFchannelsImplementation = 1	#apply separate CNN per RFfiltersFeatureTypeList #apply single CNN across all RF filters using channels (high GPU capacity required)	#not currently supported
		#RFuseParallelProcessedCNNRFchannelsImplementation = 3	#apply separate CNN per RFfilter	#not currently supported
	RFsaveRFimageSegments = False
else:
	ensureMinimumImageSizeGreaterThanRFsize = False	#CHECKTHIS
	RFsaveRFimageSegments = True
	
#****** ATORpt_RFmain ***********

np.set_printoptions(threshold=sys.maxsize)

generateRFfiltersEllipse = True
generateRFfiltersTri = False

debugLowIterations = False
debugVerbose = False
RFsaveRFfiltersAndImageSegments = True

resolutionIndexFirst = 0
numberOfResolutions = 4

imageSizeBase = (256, 256)


#****** ATORpt_RFproperties ***********

RFtypeEllipse = 1
RFtypeTri = 2
RFfeatureTypeIndexEllipse = 0	#index in list
RFfeatureTypeIndexTri = 1	#index in list

RFfeatureTypeEllipse = 1
RFfeatureTypeCircle = 2
RFfeatureTypePoint = 3
RFfeatureTypeCorner = 4

maximumAxisLengthMultiplierDefault = 4

minimumEllipseAxisLength = 1	#1
receptiveFieldOpponencyAreaFactorEllipse = 2.0	#the radius of the opponency/negative (-1) receptive field compared to the additive (+) receptive field

lowResultFilterPosition = True

#supportFractionalRFdrawSize = False	#floats unsupported by opencv ellipse draw - requires large draw, then resize down (interpolation)


#****** ATORpt_RFdetectEllipses ***********

ellipseResolutionIndexFirst = 1	#CHECKTHIS: ellipseResolutionIndexFirst = resolutionIndexFirst?
ellipseNumberOfResolutions = 6	#x; lowest res sample: 1/(2^x)	#CHECKTHIS: ellipseNumberOfResolutions = numberOfResolutions?
ellipseMinimumEllipseAxisLength = 2	#CHECKTHIS: ellipseMinimumEllipseAxisLength = minimumEllipseAxisLength?
ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 10	#degrees
ellipseColourResolution = 64	#bits


#****** ATORpt_RFellipseProperties ***********

ellipseAngleResolution = 10	#degrees
minimumEllipseFitErrorRequirement = 1500.0	#calibrate


#****** ATORpt_RFfilter ***********

minimumFilterRequirement = 1.5  # CHECKTHIS: calibrate  # matched values fraction  # theoretical value: 0.95

# if(RFsaveRFfiltersAndImageSegments):
RFfilterImageTransformFillValue = 0.0


#****** ATORpt_RFoperations ***********
opencvVersion = 3  # or 4

storeRFfiltersValuesAsFractions = True  # store RFfilters values as fractions (multipliers) rather than colours (additive)

rgbMaxValue = 255.0
rgbNumChannels = 3


#****** ATORpt_RFellipse ***********

ellipseAngleResolution = 10  # degrees
ellipseMinimumFitErrorRequirement = 1500.0  # calibrate

# match ATORpt_RFellipse algorithm;
ellipseRFnormaliseLocalEquilateralTriangle = True

ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0
ellipseNormalisedAxesLength = 1.0

# ellipse axesLength definition (based on https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69)
#  ___
# / | \ ellipseMinimumAxisLength1 | (axis1: axes width)
# | --| ellipseMinimumAxisLength2 - (axis2: axis height)
# \___/
#
ellipseMinimumAxisLength1 = minimumEllipseAxisLength * 2  # minimum elongation is required	# can be set to 1 as (axesLength1 > axesLength2) condition is enforced for RFellipse creation
ellipseMinimumAxisLength2 = minimumEllipseAxisLength
if lowResultFilterPosition:
	ellipseAxesLengthResolution = 2  # pixels (at resolution r)	# use course grain resolution to decrease number of filters # OLD: 1
else:
	ellipseAxesLengthResolution = 1  # pixels (at resolution r)
ellipseAngleResolution = 30  # degrees
ellipseColourResolution = 64  # bits


#****** ATORpt_RFtri ***********

debugSmallIterations = False

pointFeatureRFinsideRadius = 0.5
pointFeatureRFopponencyAreaFactor = 2

generatePointFeatureCorners = True
if generatePointFeatureCorners:
	triMinimumCornerOpponencyPosition = -1
	triMaximumCornerOpponencyPosition = 1
	triCornerOpponencyPositionResolution = 1
else:
	triMinimumCornerOpponencyPosition = 0
	triMaximumCornerOpponencyPosition = 0
	triCornerOpponencyPositionResolution = 1

triMatchRFellipseAlgorithm = False
triRFnormaliseLocalEquilateralTriangle = True
triNormalisedAngle = 0.0
triNormalisedCentreCoordinates = 0.0
triNormalisedAxesLength = 1.0

if triMatchRFellipseAlgorithm:
	receptiveFieldOpponencyAreaFactorTri = ATORpt_RFproperties.receptiveFieldOpponencyAreaFactorEllipse
	triMaximumAxisLengthMultiplierTri = 1
	triMaximumAxisLengthMultiplier = maximumAxisLengthMultiplierDefault
	triMinimumAxisLength1 = minimumEllipseAxisLength * 2
	triMinimumAxisLength2 = minimumEllipseAxisLength
else:
	receptiveFieldOpponencyAreaFactorTri = 1.0
	triMaximumAxisLengthMultiplierTri = 2
	triMaximumAxisLengthMultiplier = maximumAxisLengthMultiplierDefault * triMaximumAxisLengthMultiplierTri
	triMinimumAxisLength1 = minimumEllipseAxisLength * 2 * triMaximumAxisLengthMultiplierTri
	triMinimumAxisLength2 = minimumEllipseAxisLength * triMaximumAxisLengthMultiplierTri

if lowResultFilterPosition:
	triAxesLengthResolution = 1 * triMaximumAxisLengthMultiplierTri
	if debugSmallIterations:
		triAxesLengthResolution = triAxesLengthResolution * 2
else:
	triAxesLengthResolution = 1 * triMaximumAxisLengthMultiplierTri
triAngleResolution = 30
triColourResolution = 64



def printe(str):
	print(str)
	exit()

