"""ATORpt_RFmainSA.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
source activate pytorch3d
python ATORpt_RFmainSA.py images/leaf1.png

source activate sam2
python ATORpt_RFmainSA.py images/leaf1.png

# Description:
Perform ATOR receptive field (RF) ellipse detection using segment-anything (SA) library (hardware accelerated) rather than RF filters.

ATORpt RF is a receptive field implementation for ATOR feature/poly detection (ellipse centroids).

ATOR RF currently contains its own unique implementation stack, although RF feature detection can be merged into the main code base.

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available

Ellipse features/components are detected using the segment-anything (SA) library.

Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)


"""

import sys
import os
import cv2
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from ATORpt_RFglobalDefs import *
import ATORpt_RFellipsePropertiesClass
import ATORpt_RFoperations
import ATORpt_RFapplyFilter

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

if(RFuseSegmentAnything2):
	from sam2.build_sam import build_sam2
	from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator	#https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
	sam_model = build_sam2(segmentAnything2ViTHSAMcfg, segmentAnything2ViTHSAMcheckpoint)
	mask_generator = SAM2AutomaticMaskGenerator(sam_model)
else:
	from segment_anything import sam_model_registry, SamPredictor
	from segment_anything import SamAutomaticMaskGenerator
	sam = sam_model_registry[segmentAnythingViTHSAMname](checkpoint=segmentAnythingViTHSAMpathName)
	sam.to(device)
	predictor = SamPredictor(sam)
	mask_generator = SamAutomaticMaskGenerator(sam)


def main(image_path):
	return generateATORRFpatchesImage(image_path)

def generateATORpatches(use3DOD, imagePaths, train):

	if(use3DOD):
		printe("generateATORpatches error: use3DOD not currently supported")
		
	transformedPatchesList = []
	for imageIndex, imagePath in enumerate(imagePaths):
		transformedPatches = generateATORRFpatchesImage(imagePath)
		transformedPatchesList.append(transformedPatches)
	transformedPatches = pt.stack(transformedPatchesList, dim=0)	#shape: batchSize, numberEllipses, H, W, C
	transformedPatches = pt.permute(transformedPatches, (0, 1, 4, 2, 3))	#shape: batchSize, VITmaxNumberATORpatches, C, H, W
	
	return transformedPatches
		
def generateATORRFpatchesImage(image_path):
	# Read image
	image_rgb = read_image(image_path)

	resolutionIndexMax = numberOfResolutions
	if RFscaleImage:
		resolutionIndexMin = resolutionIndexFirst
	else:
		resolutionIndexMin = resolutionIndexMax
	
	ellipsePropertiesListAllRes = []
	for resolutionIndex in range(resolutionIndexMin, resolutionIndexMax+1):
		ellipsePropertiesList = detectRFs(image_rgb, resolutionIndex)
		ellipsePropertiesListAllRes = ellipsePropertiesListAllRes + ellipsePropertiesList
	
	transformedPatchList = []
	for ellipse in ellipsePropertiesListAllRes:
		transformedPatches = generateNormalisedImageSegment(ellipse, image_rgb)
		for transformedPatch in transformedPatches:
			transformedPatch = pt.tensor(transformedPatch)	#convert np to pt
			transformedPatchList.append(transformedPatch)
	transformedPatches = pt.stack(transformedPatchList, dim=0)	#shape: numberEllipses, H, W, C

	numberEllipses = transformedPatches.shape[0]	#len(ellipsePropertiesListAllRes) 
	#print("numberEllipses = ", numberEllipses)
	if(numberEllipses < VITmaxNumberATORpatches):
		paddedPatchesNumber = VITmaxNumberATORpatches-numberEllipses
		paddedPatches = pt.zeros((paddedPatchesNumber, transformedPatches.shape[1], transformedPatches.shape[2], transformedPatches.shape[3]))
		transformedPatches = pt.concat((transformedPatches, paddedPatches), dim=0)
	elif(numberEllipses > VITmaxNumberATORpatches):
		print("generateATORRFpatchesImage warning: numberEllipses > VITmaxNumberATORpatches, remove excess patches: numberEllipses = ", numberEllipses, ", VITmaxNumberATORpatches = ", VITmaxNumberATORpatches)
		transformedPatches = transformedPatches[0:VITmaxNumberATORpatches]
	elif(numberEllipses == VITmaxNumberATORpatches):
		transformedPatches = transformedPatches	#no change
		
	return transformedPatches	
		

def detectRFs(image_rgb, resolutionIndex):
	inputImageHeight, inputImageWidth, inputImageChannels = image_rgb.shape
	imageSizeBase = (inputImageWidth, inputImageHeight)
	
	if(debugVerbose):
		draw_image(image_rgb, "original image")

	resolutionProperties = ATORpt_RFoperations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase)
	#print("resolutionIndex = ", resolutionIndex)
	#print("resolutionProperties.resolutionFactor = ", resolutionProperties.resolutionFactor)
	#print("resolutionProperties.imageSize = ", resolutionProperties.imageSize)

	resizedImage = cv2.resize(image_rgb, resolutionProperties.imageSize, interpolation=cv2.INTER_LINEAR)

	# Detect segments
	features = detect_segments(resizedImage)

	# Detect ellipses
	ellipsePropertiesList = detect_ellipses(features, resolutionProperties)
	
	if(debugVerbose):
		# Draw original image + outline
		draw_original_image_and_outline(resizedImage, features)

	return ellipsePropertiesList
	
def read_image(image_path):
	"""
	2. Read an image file into memory (BGR with OpenCV).
	   Convert to RGB for consistent processing/visualization.
	"""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image file not found: {image_path}")

	image_bgr = cv2.imread(image_path)
	if image_bgr is None:
		raise ValueError(f"Could not read image from: {image_path}")

	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	return image_rgb

def compute_contrast_map(image_rgb, method='laplacian', ksize=3):
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
	if method == 'laplacian':
		# Apply Laplacian filter to detect edges/contrast
		image_contrast = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=ksize)
		image_contrast = np.abs(image_contrast)  # Take absolute value
	elif method == 'std':
		# Compute local standard deviation for contrast estimation
		blurred = cv2.GaussianBlur(image_gray, (ksize, ksize), 0)
		image_contrast = np.sqrt(cv2.GaussianBlur(image_gray**2, (ksize, ksize), 0) - blurred**2)
	else:
		raise ValueError("Invalid method. Choose 'laplacian' or 'std'.")
	# Normalize to 0-255
	image_contrast = cv2.normalize(image_contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return image_contrast
	
def detect_segments(image_rgb):

	height, width, _ = image_rgb.shape
	
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
	image_contrast = compute_contrast_map(image_rgb, method='laplacian')
	
	# use Segment Anything to get segment masks
	masks = []

	if(RFuseSegmentAnything2):
		masks = mask_generator.generate(image_rgb)
	else:
		predictor.set_image(image_rgb)
		masks = mask_generator.generate(image_rgb)	
	
	segment_points = []
	edge_points = []
	centroid_points = []
	colour_points = []

	if len(masks) == 0:
		print("detect_segments error: no SAM masks found or checkpoint missing.")
	else:
		for m in masks:
			# m is a dict with keys 'segmentation', 'area', 'bbox', ...
			seg = m["segmentation"].astype(np.uint8)
		
			# Colour detection
			mask = seg.astype(bool)
			colour_segment = [image_rgb[:, :, c][mask].mean() for c in range(3)]  # R, G, B order
			
			segFilterPass = True
			if(RFfilterSegments):
				image_area = height*width
				mask_area = m["segmentation"].sum()
				mask_ratio = mask_area / image_area
				#print("mask_ratio = ", mask_ratio)
				colour_segment_lum = np.mean(colour_segment)
				#print("colour_segment_lum = ", colour_segment_lum)
				if mask_ratio > RFfilterSegmentsWholeImageThreshold:
					segFilterPass = False
					#print("mask_ratio = ", mask_ratio)
				if colour_segment_lum < RFfilterSegmentsBackgroundColourThreshold:
					segFilterPass = False
					#print("colour_segment_lum = ", colour_segment_lum)

			if(segFilterPass):
				# Edge detection - find the boundary of the segment (contour)
				contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				edge_points_segment = []
				for c in contours:	#should only be one contour per segment
					for pt in c:
						ix = pt[0][0]	#x
						iy = pt[0][1]	#y
						if ix < 0 or ix >= width or iy < 0 or iy >= height:
							continue
						contrast = float(image_contrast[iy, ix])  # simplistic contrast measure
						edge_points_segment.append((ix, iy, contrast))
				edge_points_segment_np = np.array(edge_points_segment, dtype=np.float32)
				edge_points.append(edge_points_segment_np)

				# Colour detection
				colour_points.append(colour_segment)

				# Segment coordinates derivation
				segment_points_segment = np.argwhere(seg > 0)  # Returns (row, col) -> (y, x)
				segment_points_segment = segment_points_segment[:, [1, 0]]  # Swap columns to get (x, y)
				segment_points.append(segment_points_segment)

				# Centroid detection
				M = cv2.moments(seg, binaryImage=True)
				if M["m00"] != 0:
					cx = M["m10"] / M["m00"]
					cy = M["m01"] / M["m00"]
					# get contrast from the center pixel
					ix, iy = int(round(cx)), int(round(cy))
					if ix >= 0 and ix < width and iy >= 0 and iy < height:
						contrast = float(image_contrast[iy, ix])
						centroid_points.append((cx, cy, contrast))

	features = {
		"segment_points": segment_points,
		"edge_points": edge_points,
		"centroid_points": centroid_points,
		"colour_points": colour_points,
	}
	
	return features

def detect_ellipses(features, resolutionProperties):
	segment_points = features["segment_points"]
	colour_points = features["colour_points"]
	ellipsePropertiesList = []
	
	for segment_index, segment_points_segment in enumerate(segment_points):
		colour = colour_points[segment_index]
		if len(segment_points_segment) >= 5:

			ellipse = cv2.fitEllipse(segment_points_segment.astype(np.float32))
			centerCoordinates, axesLength, angle = rescaleEllipseCoordinates(ellipse, resolutionProperties)
			ellipseProperties = ATORpt_RFellipsePropertiesClass.EllipsePropertiesClass(centerCoordinates, axesLength, angle, colour)
			#inputImageRmod, ellipseFitError = ATORpt_RFellipsePropertiesClass.testEllipseApproximation(inputImageR, ellipseProperties)
			ellipsePropertiesList.append(ellipseProperties)

			if(debugVerbose):
				print("\nEllipse #", segment_index)
				(cx, cy) = centerCoordinates
				(majorAxis, minorAxis) = axesLength
				print(f"Center: ({cx:.2f}, {cy:.2f})")
				print(f"Major Axis: {majorAxis:.2f}")
				print(f"Minor Axis: {minorAxis:.2f}")
				print(f"Rotation Angle: {angle:.2f}")
				print("colour = ", colour)

	return ellipsePropertiesList

def rescaleEllipseCoordinates(ellipse, resolutionProperties):
	centerCoordinates, axesLength, angle = ellipse

	if(RFalwaysDefineEllipseMajorAxisAsFirst):
		#correct for cv2.fitEllipse discrepancy;
		# Ensure majorAxis is always greater than minorAxis
		majorAxis, minorAxis = max(axesLength), min(axesLength)
		# Adjust the angle if OpenCV mislabels the axes
		if axesLength[0] > axesLength[1]:  # If OpenCV got it right, keep angle as is
			angle = angle
		else:  # If OpenCV swapped major/minor, adjust by 90 degrees
			angle = (angle + 90) % 180
	else:
		(majorAxis, minorAxis) = axesLength
		
	(cx, cy) = centerCoordinates
	cx = cx*resolutionProperties.resolutionFactor
	cy = cy*resolutionProperties.resolutionFactor
	majorAxis = majorAxis*resolutionProperties.resolutionFactor
	minorAxis = minorAxis*resolutionProperties.resolutionFactor
	centerCoordinates = (cx, cy)
	axesLength = (majorAxis, minorAxis)
		
	return centerCoordinates, axesLength, angle

def draw_original_image_and_outline(image_rgb, features):

	segment_points = features["segment_points"]
	edge_points = features["edge_points"]
	centroid_points = features["centroid_points"]
	centroid_points = np.array(centroid_points, dtype=np.float32)
	
	plt.figure("Step 5 - Original + Outline", figsize=(8, 6))
	plt.imshow(image_rgb)
	plt.title("Original Image + Mesh Outline")
	
	for segment_points_segment in segment_points:
		plt.scatter(segment_points_segment[:,0], segment_points_segment[:,1], c="blue", s=1)
	for edge_points_segment in edge_points:
		plt.scatter(edge_points_segment[:,0], edge_points_segment[:,1], c="green", s=2)
	plt.scatter(centroid_points[:,0], centroid_points[:,1], c="red", s=3)

	plt.axis("off")
	plt.show(block=True)  # block=False so we can continue
	
def calculateEllipticity(ellipse):
	axesLength = ellipse.axesLength
	a, b = max(axesLength), min(axesLength)
	if a <= 0 or b <= 0:
		raise ValueError("Semi-major and semi-minor axes must be positive numbers.")
	if b > a:
		raise ValueError("Semi-major axis (a) must be greater than or equal to semi-minor axis (b).")
	e = math.sqrt(1 - (b ** 2) / (a ** 2))
	return e
	
def generateNormalisedImageSegment(ellipse, image_rgb):
	patch_transformed_list = []
	
	patch, patch_topleft = ATORpt_RFapplyFilter.crop_ellipse_area(image_rgb, ellipse, padding_ratio=1.0)
	if(debugVerbose):
		ATORpt_RFapplyFilter.draw_patch_with_ellipse(patch, ellipse, patch_topleft, title_str="patch_orig")

	if(RFdetectEllipticity):
		e = calculateEllipticity(ellipse)
		if(debugVerbose):
			print("calculateEllipticity(ellipse) = ", e)
		if(e > RFminimumEllipticityThresholdRotate):
			patch_transformed = ATORpt_RFapplyFilter.transform_patch(patch, ellipse, patch_topleft)
			patch_transformed_list.append(patch_transformed)
			if(debugVerbose):
				ATORpt_RFapplyFilter.draw_patch_with_circle(patch_transformed, center=(RFpatchCircleOffset,RFpatchCircleOffset), diameter=RFpatchCircleWidth, title_str="patch_transformed")
		if(e < RFmaximumEllipticityThresholdNoRotate):
			patch_transformed = ATORpt_RFapplyFilter.transform_patch_scale_only(patch, ellipse, patch_topleft)
			patch_transformed_list.append(patch_transformed)
			if(debugVerbose):
				ATORpt_RFapplyFilter.draw_patch_with_circle(patch_transformed, center=(RFpatchCircleOffset,RFpatchCircleOffset), diameter=RFpatchCircleWidth, title_str="patch_transformed")
	else:
		patch_transformed = ATORpt_RFapplyFilter.transform_patch(patch, ellipse, patch_topleft)
		patch_transformed_list.append(patch_transformed)
		if(debugVerbose):
			ATORpt_RFapplyFilter.draw_patch_with_circle(patch_transformed, center=(RFpatchCircleOffset,RFpatchCircleOffset), diameter=RFpatchCircleWidth, title_str="patch_transformed")
	
	return patch_transformed_list

def draw_image(patch_rgb, name):	
	plt.figure(name, figsize=(8, 6))
	plt.imshow(patch_rgb)
	plt.title(name)
	plt.axis("off")
	plt.show(block=True)  # block=False so we can continue
	

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python ATORpt_RFmainSA.py <input_image>")
		sys.exit(1)
	input_image_path = sys.argv[1]
	
	main(input_image_path)

