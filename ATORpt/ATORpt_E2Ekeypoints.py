"""ATORpt_E2Ekeypoints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E keypoints

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *
import ATORpt_E2Eoperations

def performKeypointDetection(self, featureMapN, posEmbeddings, posEmbeddingsNormalised, geometricHashingPixelPosEmbeddings):

	#proximity calculations;

	#https://stackoverflow.com/questions/29063851/how-to-parallelized-scipy-cosine-similarity-calculation
	posEmbeddings1 = posEmbeddings/pt.linalg.norm(posEmbeddings, axis=1)[:,None]
	posEmbeddings1 = pt.nan_to_num(posEmbeddings1, nan=0.0)  #CHECKTHIS; nan=0.0

	dotsProximity = pt.einsum('ik,jk->ij', posEmbeddings1, posEmbeddings1)

	dotsLuminosity = featureMapN
	#print("featureMapN.shape = ", featureMapN.shape)
	#print("dotsLuminosity.shape = ", dotsLuminosity.shape)
	#print("dotsProximity.shape = ", dotsProximity.shape)
	dots = pt.multiply(dotsLuminosity, dotsProximity)

	#if(debugGeometricHashingParallel):
	#	ATORpt_E2Eoperations.printFeatureMap(posEmbeddings, dots)

	if(useGeometricHashingProbabilisticKeypoints):
		attention = dots
		if(useGeometricHashingProbabilisticKeypointsSoftMax):
			attention = self.softmax(attention / (self.numberOfTokenDimensions ** 0.5))   #attention = self.softmax(dots)
		if(useGeometricHashingProbabilisticKeypointsNonlinearity):
			attention = self.offsetReLU(attention) #apply non-linearity to select small number of keypoints	#self.activationFunction(dots)

		attention = pt.unsqueeze(attention, dim=2)

		posEmbeddingsRepeat = posEmbeddingsNormalised
		posEmbeddingsRepeat = pt.unsqueeze(posEmbeddingsRepeat, dim=1)
		numRep = posEmbeddingsRepeat.shape[0]
		posEmbeddingsRepeat = posEmbeddingsRepeat.repeat(1, posEmbeddingsRepeat.shape[0], 1)   

		if(useGeometricHashingProbabilisticKeypointsZero):
			attentionThresholded = pt.gt(attention, 0.0).type(pt.float) 
			posEmbeddingsRepeat = pt.multiply(posEmbeddingsRepeat, attentionThresholded)

		geometricHashingKeypointsPosEmbeddings = pt.cat([posEmbeddingsRepeat, attention], dim=2)

		pixelPosEmbeddingAttentionArtificial = pt.ones(1,1).repeat(sequenceLength, 1)
		geometricHashingPixelPosEmbeddings = pt.cat([geometricHashingPixelPosEmbeddings, pixelPosEmbeddingAttentionArtificial], dim=1)	#set pixel embedding artificial attention value to 1.0 (this is only used to ensure that AMANN input is consistent/even)
	else:
		keypoints = pt.topk(dots, k=self.geometricHashingNumKeypoints, dim=1)
		keypointsIndices = keypoints.indices
		keypointsValues = keypoints.values

		keypointsIndicesFlattened = pt.reshape(keypointsIndices, (keypointsIndices.shape[0]*keypointsIndices.shape[1],))  #or flatten	#keypointsIndicesFlattened = keypointsIndices.flatten()
		keypointsPosEmbeddingsFlattened = posEmbeddingsNormalised[keypointsIndicesFlattened]
		keypointsPosEmbeddings = pt.reshape(keypointsPosEmbeddingsFlattened, (keypointsIndices.shape[0], keypointsIndices.shape[1], self.numberOfGeometricDimensions))  #CHECKTHIS
		geometricHashingKeypointsPosEmbeddings = keypointsPosEmbeddings

	return geometricHashingKeypointsPosEmbeddings, geometricHashingPixelPosEmbeddings


def offsetReLU(self, Z):
	Z = Z - useGeometricHashingProbabilisticKeypointsNonlinearityOffset
	A = self.activationFunction(Z)
	return A

