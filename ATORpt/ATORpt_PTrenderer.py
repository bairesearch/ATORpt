"""ATORpt_PTrenderer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

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
import ATORpt_operations

import torch as pt
import torch.nn.functional as F

import matplotlib.pyplot as plt
if(snapshotRenderer == "pytorch3D"):
	from pytorch3d.structures import Meshes
	from pytorch3d.renderer import OpenGLPerspectiveCameras, MeshRenderer, MeshRasterizer, TexturesVertex, RasterizationSettings, SoftPhongShader, Textures
	#from pytorch3d.renderer.mesh.shader import LambertShader
	from pytorch3d.renderer.cameras import look_at_view_transform, FoVPerspectiveCameras, OrthographicCameras, FoVOrthographicCameras
	from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
elif(snapshotRenderer == "torchgeometry"):
	import torchgeometry as tgm

def resamplePixelCoordinates(transformedSnapshotPixelCoordinates, snapshotMeshValues, snapshotMeshFaces, renderImageSize):
	if(snapshotRenderer == "pytorch3D"):
		transformedPatches = renderSnapshotsPytorch3D(transformedSnapshotPixelCoordinates, snapshotMeshFaces, snapshotMeshValues, renderImageSize)
	elif(snapshotRenderer == "torchgeometry"):
		transformedPatches = renderSnapshotsTorchGeometry(transformedSnapshotPixelCoordinates, snapshotMeshFaces, snapshotMeshValues, renderImageSize)
	return transformedPatches
	
def renderSnapshotsTorchGeometry(verts, colors, renderImageSize):
	polygons_vertices = verts	#polygons_vertices = pt.reshape(verts, (verts.shape[0], verts.shape[1]//4, 4, 2))	#polygons_vertices = torch.randint(0, 256, (batch_size, num_polygons_per_image, 4, 2)) 
	polygons_colors = colors.unsqueeze(-1).unsqueeze(-1)	#torch.randint(0, 256, (batch_size, num_polygons_per_image, 3, 1, 1))  # RGB colors for each polygon, broadcastable to image size
	print("polygons_vertices.shape = ", polygons_vertices.shape)
	print("polygons_colors.shape = ", polygons_colors.shape)
	polygons_vertices = polygons_vertices.unsqueeze(2).expand(-1, -1, 3, -1, -1)
	polygons_colors = polygons_colors.unsqueeze(3).expand(-1, -1, 3, *renderImageSize)
	print("polygons_vertices.shape = ", polygons_vertices.shape)
	print("polygons_colors.shape = ", polygons_colors.shape)
	masks = tgm.fill_convex_polyfill(images, polygons_vertices, value=1)
	images = masks * polygons_colors

	images = images.sum(dim=1)
	
	if(snapshotRenderDebug):
		printImages1(images)
		
	return images
	
def printImages1(images):
	plt.imshow(images[0].permute(1, 2, 0).numpy().astype(int))
	plt.show()
	
	ex

def renderSnapshotsPytorch3D(verts, faces, colors, renderImageSize):
	origImageSize = 300
	
	numSnapshots = verts.shape[0]
	numCameras = numSnapshots
	
	vertsZ = pt.ones((verts.shape[0], verts.shape[1], 1)).to(device)*snapshotRenderZdimVal	#add Z dimension
	verts = pt.cat((verts, vertsZ), dim=2)
	
	print("verts.shape = ", verts.shape)
	print("faces.shape = ", faces.shape)
	print("colors.shape = ", colors.shape)

	#textures = TexturesVertex(verts_features=colors.to(device))
	textures = Textures(verts_rgb=colors.to(device)) 
	#textures = Textures(verts_rgb=pt.rand(verts.shape[0], verts.shape[1], 3)) 
	meshes = Meshes(verts=verts, faces=faces, textures=textures)
	
	#default values for R/T: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.FoVOrthographicCameras
	#R = pt.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
	#T = pt.tensor([[0., 0., 0]]) 
	
	#R, T = look_at_view_transform(dist=100, elev=15., azim=0., device=device), 
	'''
	theta = pt.tensor(10)  # 90 degrees
	theta_rad = pt.deg2rad(theta)
	cos_theta = pt.cos(theta_rad)
	sin_theta = pt.sin(theta_rad)
	Rz = pt.tensor([[[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]])
	'''
	
	cameras = FoVOrthographicCameras(min_x=0, min_y=0, max_x=origImageSize, max_y=origImageSize, znear=0.1, zfar=100.0, device=device)	#R=R	R=R, T=T
	'''
	R = cameras.R
	print("R = ", R)
	#R = pt.matmul(rotation_matrix, R)
	print("Rz = ", Rz)
	#cameras.R = Rz
	'''
	
	shader = SoftPhongShader(device=device, cameras=cameras, lights=None)
	rasterizer = MeshRasterizer(cameras=cameras)
	renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
	images = renderer(meshes)	#TODO: CUDA out of memory
	
	#print("verts[0] = ", verts[0])
	#print("faces[0] = ", faces[0])
	#print("colors[0] = ", colors[0])
	#print("images[0] = ", images[0])
	
	images = images[..., :3]	#remove the alpha channel
	
	if(snapshotRenderDebug):
		printImages3(images)
	
	return images

def printImages3(images):
	for image in images:
		#print("image = ", image)
		plt.figure(figsize=(8, 8))
		plt.imshow(image.squeeze().cpu().numpy())
		plt.axis('off')
		plt.show()
	
