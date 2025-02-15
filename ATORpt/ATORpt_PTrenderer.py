"""ATORpt_PTrenderer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT renderer

"""

import torch as pt
import math

from ATORpt_globalDefs import *

import torch as pt
import torch.nn.functional as F

import matplotlib.pyplot as plt
if(snapshotRenderer == "pytorch3D"):
	from pytorch3d.structures import Meshes
	from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesVertex, Textures, RasterizationSettings
	from pytorch3d.renderer.mesh.shader import HardFlatShader
	from pytorch3d.renderer.cameras import FoVOrthographicCameras
	from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
	from pytorch3d.renderer import PointLights, Materials

def resamplePixelCoordinates(use3DOD, transformedSnapshotPixelCoordinates, snapshotMeshValues, snapshotMeshFaces, renderViewportSize, renderImageSize, centreSnapshots=False, index=None):
	if(snapshotRenderer == "pytorch3D"):
		transformedPatches = renderSnapshotsPytorch3D(use3DOD, transformedSnapshotPixelCoordinates, snapshotMeshFaces, snapshotMeshValues, renderViewportSize, renderImageSize, centreSnapshots, index)
	return transformedPatches
	
def renderSnapshotsPytorch3D(use3DOD, verts, faces, colors, renderViewportSize, renderImageSize, centreSnapshots=False, index=None):
	#print("verts = ", verts)
	if(not use3DOD):
		#add Z dimension to coordinates
		vertsZ = pt.ones((verts.shape[0], verts.shape[1], 1)).to(device)
		vertsZ = vertsZ*snapshotRenderZdimVal
		verts = pt.cat((verts, vertsZ), dim=2)	#zAxisGeometricHashing
	
	textures = TexturesVertex(verts_features=colors.to(device))		#textures = Textures(verts_rgb=colors.to(device)) 
	meshes = Meshes(verts=verts, faces=faces, textures=textures)
	
	T = pt.tensor([[0., 0., 0.]])
	angle_z = pt.deg2rad(pt.tensor(snapshotRenderCameraRotationZaxis))
	angle_y = pt.deg2rad(pt.tensor(snapshotRenderCameraRotationYaxis))
	angle_x = pt.deg2rad(pt.tensor(snapshotRenderCameraRotationXaxis))
	R = euler_angles_to_matrix(pt.tensor([angle_z, angle_y, angle_x]), "ZYX")
	R = R.unsqueeze(dim=0)
	
	min_x=-renderViewportSize[xAxisGeometricHashing]/2
	max_x=renderViewportSize[xAxisGeometricHashing]/2
	if(centreSnapshots):
		min_y=-renderViewportSize[yAxisGeometricHashing]/2
		max_y=renderViewportSize[yAxisGeometricHashing]/2
		#or T = pt.tensor([[0., 0., 0.]])
	else:
		#according to ATOR 2D0D specification, object triangles (ie normalised snapshots) are centred wrt the y axis, and placed on (ie above) the x axis
		if(renderInvertedYaxisToDisplayOriginalImagesUpright):
			min_y=-renderViewportSize[yAxisGeometricHashing]
			max_y=0
		else:
			min_y=0
			max_y=renderViewportSize[yAxisGeometricHashing]
		#or T = pt.tensor([[0., -renderViewportSize[yAxisGeometricHashing]//2., 0.]])
	if(use3DOD):
		if(ATOR3DODgeoHashingAlignObjectTriangleBaseVertically):
			#object triangle base is aligned with y axis (rather than x axis)
			min_yTemp, max_yTemp = min_y, max_y
			min_y, max_y = min_x, max_x
			min_x, max_x = min_yTemp, max_yTemp
		#ATOR 3DOD: expects 

	cameras = FoVOrthographicCameras(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, znear=snapshotRenderCameraZnear, zfar=snapshotRenderCameraZfar, R=R, T=T, device=device)

	lights = PointLights(device=device, ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
	shader = HardFlatShader(device=device, cameras=cameras, lights=lights)
	
	raster_settings = RasterizationSettings(image_size=renderImageSize, blur_radius=0.0, faces_per_pixel=1)
	rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
	renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
	images = renderer(meshes)	#format: N, H, W, C	#CUDA will throw out of memory if too many polys (see ATORmaxNumberATORpatchesAllImages)
	
	images = images[..., :3]	#remove the alpha channel

	if(applyObjectTriangleMask):
		images = maskImageCoordiantesOutsideObjectTriangle(images, renderImageSize)
			
	if(debugSnapshotRender):
		if(index is not None):
			title = "poly index: " + str(index)
			printImage(images[index], title=title)
		else:
			printImages(images)
	elif(debugSnapshotRenderFullImage):
		print("debugSnapshotRenderFullImage:")
		printImages(images)
	elif(debugSnapshotRenderFinal):
		print("debugSnapshotRenderFinal:")
		printImages(images)
	
	return images

def printImages(images):
	for index, image in enumerate(images):
		#print("printImages: index = ", index)
		title = "poly index: " + str(index)
		printImage(image, title=title)

def printImage(image, title=""):
	#print("image = ", image)
	fig = plt.figure(figsize=(8, 8), facecolor='lightgray')
	fig.canvas.manager.set_window_title(title)
	plt.imshow(image.squeeze().cpu().numpy())
	plt.axis('off')
	plt.show()

def maskImageCoordiantesOutsideObjectTriangle(images, renderImageSize):
	mask = createObjectTriangleMask(renderImageSize)
	if(debugSnapshotRender):
		#only draw outline of object triangle
		maskInside = createObjectTriangleMask(renderImageSize, diagonal=-1)
		maskInside = pt.logical_not(maskInside)
		mask =  pt.logical_and(mask, maskInside)
		mask =  pt.logical_not(mask)	#create black tri outline
	mask = mask.unsqueeze(0).unsqueeze(-1)
	images = images*mask
	return images

def createObjectTriangleMask(size, diagonal=0):
	if(normalisedObjectTriangleHeight == 1):
		#renderImageSize = 3
		mask1 = pt.ones(size//2, size//2).to(device)
		mask1 = pt.tril(mask1, diagonal)
		mask1 = pt.repeat_interleave(mask1, 2, dim=0)
		mask2 = pt.flip(mask1, [1])
		mask = pt.cat((mask2, mask1), dim=1)
		if(renderInvertedYaxisToDisplayOriginalImagesUpright):
			mask = pt.flip(mask, [0])
		return mask
	else:
		printe("createObjectTriangleMask currently requires (normalisedObjectTriangleHeight == 1)")

