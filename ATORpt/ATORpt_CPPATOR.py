"""ATORpt_CPPATOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt CPP ATOR - interface for ATOR C++ implementation: transformed ("normalised") image patch generation

"""

import torch as pt
import os
import subprocess
from PIL import Image
from torchvision import transforms

from ATORpt_PTglobalDefs import *

supportFolderGeneration = False	#rely on ATOR cpp implementation to generate folder structure


def generateATORpatches(imagePaths, train):
	imagePath = imagePaths[0]
	assert len(imagePaths) == 1
		
	copyImage(imagePath, inputfolder)	#copy image to ATOR input folder
	inputImageName = getFileName(imagePath)
	print("imagePath = ", imagePath)
	print("inputImageName = ", inputImageName)
	inputImageWidth, inputImageHeight, inputObjectName = getATORinputImageSize(inputImageName)
	ATORexePath = exeFolder + ATORCexe
	command = [ATORexePath, '-trainortest', '1', '-object', inputObjectName, '-imageext', '.png', '-width', str(inputImageWidth), '-height', str(inputImageHeight), '-cleartrain', '-inputfolder', inputfolder, '-exefolder', exefolder]
	try:
		print("ATOR.exe -trainortest 1 -object house2DOD -imageext .png -width 768 -height 576 -cleartrain")
		subprocess.run(command, check=True)
		print("Success executing command")
	except subprocess.CalledProcessError as e:
		print("Error executing command", e)
	imageTensors = loadATORpatches(inputImageName, train)
	return imageTensors

def loadATORpatches(inputImageName, train):	
	objectName = getImageObjectName(inputImageName)
	folderPath = DBgenerateFolderName(objectName, train)
	patchIndex = 0
	imageTensorList = []
	for polyIndex in range(VITmaxNumberATORpolysPerZoom):
		for zoomIndex in range(ATOR_METHOD2DOD_NUMBER_OF_SNAPSHOT_ZOOM_LEVELS):
			for sideIndex in range(ATOR_METHOD_POLYGON_NUMBER_OF_SIDES):
				if(patchIndex < VITmaxNumberATORpatches):
					viewIndex = 0
					polyIndex = patchIndex
					if(train):
						trainString = "train"
					else:
						trainString = "test"
					patchFileName = objectName + "interpolatedMeshViewIndex" + str(viewIndex) + "ZoomIndex" + str(zoomIndex) + "FacingPoly" + str(polyIndex) + "side" + str(sideIndex) + ".rgb." + trainString + ".ppm"
					patchPathName = folderPath + patchFileName
					if(fileExists(patchPathName)):
						imageTensor = loadImage(patchPathName)
						print("transformed patch fileExists: imageTensor.shape = ", imageTensor.shape, ", patchPathName = ", patchPathName)
						imageTensorList.append(imageTensor)
						patchIndex += 1
					#else:
					#	print("transformed patch !fileExists: imageTensor.shape = ", imageTensor.shape, ", patchPathName = ", patchPathName)
	for patchIndex in range(patchIndex, VITmaxNumberATORpatches, 1):
		imageTensor = pt.ones((VITnumberOfChannels, VITpatchSize[0], VITpatchSize[1])).to(device) * paddingPatchTokenValue
		imageTensorList.append(imageTensor)
		#add padding patches
	
	imageTensors = pt.stack(imageTensorList, dim=0)
	imageTensors.to(device)
	return imageTensors
	

def loadImage(imagePath):
	image = Image.open(imagePath)
	#image = image.convert("RGB")
	transform = transforms.ToTensor()
	imageTensor = transform(image).to(device)	#C, H, W
	return imageTensor

def getALOIVIEWImagePath(imageIndex, viewIndex):
	viewName = str(viewIndex*(360/ALOIdatabaseNumberOfViews))
	imageName = str(imageIndex) + "_r" + viewName + ".png"
	imagePath = databaseRoot + "aloi_view/png/" + str(imageIndex) + "/" + imageName
	return imagePath
		
		
def getATORinputImageSize(imageName):
	imagePath = inputfolder + "/" + imageName
	image = Image.open(imagePath)
	width, height = image.size
	imageNameWOextension = getImageObjectName(imageName)
	return width, height, imageNameWOextension

#derived from ATORdatabaseFileIO.cpp
def DBgenerateServerDatabaseName(objectName, train):
	databaseName = ATOR_DATABASE_FILESYSTEM_DEFAULT_SERVER_OR_MOUNT_NAME + ATOR_DATABASE_FILESYSTEM_DEFAULT_DATABASE_NAME
	return databaseName

#derived from ATORdatabaseFileIO.cpp
def DBgenerateFolderName(objectName, train):
	databaseName = DBgenerateServerDatabaseName(objectName, train)
	folderPath = databaseName

	if(train):
		datasetTypeFolderName = ATOR_DATABASE_TRAIN_FOLDER_NAME
	else:
		datasetTypeFolderName = ATOR_DATABASE_TEST_FOLDER_NAME
		
	if ATOR_DATABASE_TRAIN_TEST_FOLDER_STRUCTURE_SAME and not train:
		folderPath = folderPath + datasetTypeFolderName + "/"
	else:
		folderPath = folderPath + datasetTypeFolderName + "/"
		numberOfEntityNameLevels = min(len(objectName), ATOR_DATABASE_CONCEPT_NAME_SUBDIRECTORY_INDEX_NUMBER_OF_LEVELS)
		for level in range(numberOfEntityNameLevels):
			folderName = objectName[level]
			folderPath = folderPath + folderName + "/"
		folderPath = folderPath + objectName + "/"

	return folderPath


def getImageObjectName(imageName):
	imageNameWOextension = os.path.splitext(os.path.basename(imageName))[0]
	return imageNameWOextension
	
def copyImage(sourcePath, destinationPath):
	print("cp " + sourcePath + " " + destinationPath)
	subprocess.run(["cp", sourcePath, destinationPath])

def getFileName(filePath):
	fileName = os.path.basename(filePath)
	return fileName

def getCurrentFolder():
	currentFolder = os.getcwd()
	return currentFolder

def fileExists(filePath):
	if(os.path.exists(filePath)):
		result = True
	else:
		result = False
	return result

if __name__ == '__main__':
	currentFolder = getCurrentFolder()
	imageName = "house2DOD.png"
	imagePath = currentFolder + "/" + imageName
	print("imagePath = ", imagePath)
	generateATORpatches(imagePath)
