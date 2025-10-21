"""ATORpt_CPPglobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt CPP global definitions

"""

from ATORpt_globalDefs import * 


VITmaxNumberATORpolysPerZoom = VITmaxNumberATORpatches//numberOfZoomLevels	#300	#CHECKTHIS
batchSize = 1	#process images serially
useParallelisedGeometricHashing = False
usePositionalEmbeddings = False
useE2EclassificationVIT = True
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

