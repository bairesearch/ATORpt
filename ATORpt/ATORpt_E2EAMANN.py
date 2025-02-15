"""ATORpt_E2EAMANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt E2E AMANN - additive-multiplicative artificial neural network

"""

import torch as pt
import torch.nn as nn

from ATORpt_globalDefs import *


class LayerAdditiveMultiplicativeClass(nn.Module):
	def __init__(self, inputFeatures, outputFeatures, useBias=False, useMultiplicativeUnits=True):
		super().__init__()
		self.inputFeatures = inputFeatures
		self.outputFeatures = outputFeatures
		self.useBias = useBias
		self.useMultiplicativeUnits = useMultiplicativeUnits

		if(useMultiplicativeUnits):
			self.inputFeaturesAdditive = inputFeatures
			self.outputFeaturesAdditive = outputFeatures//2
			self.inputFeaturesMultiplicative = inputFeatures
			self.outputFeaturesMultiplicative = outputFeatures//2

			self.Wa = torch.nn.Parameter(pt.randn(self.inputFeaturesAdditive, self.outputFeaturesAdditive))
			self.Wm = torch.nn.Parameter(pt.randn(self.inputFeaturesMultiplicative, self.outputFeaturesMultiplicative))
			if(self.useBias):
				self.Ba = torch.nn.Parameter(pt.zeros(self.outputFeaturesAdditive))  #randn
				self.Bm = torch.nn.Parameter(pt.zeros(self.outputFeaturesMultiplicative))	#randn
		else:
			self.W = torch.nn.Parameter(pt.randn(self.inputFeatures, self.outputFeatures))
			if(self.useBias):
				self.B = torch.nn.Parameter(pt.zeros(self.outputFeatures))   #randn
 
		self.activationFunction = torch.nn.ReLU()

	def forward(self, input):
		x = input
		if(self.useMultiplicativeUnits):
			AprevLayerA = x
			AprevLayerA = self.clipActivation(AprevLayerA)
			AprevLayerM = self.multiplicativeEmulationFunctionPre(AprevLayerA)
			#print("self.Wa.shape = ", self.Wa.shape)
			#print("self.Wm.shape = ", self.Wm.shape)
			Za = AprevLayerA @ self.Wa
			Zm = AprevLayerM @ self.Wm
			Zm = self.multiplicativeEmulationFunctionPost(Zm)
			if(self.useBias):
				Za = Za + self.Ba
				Zm = Zm + self.Bm
 
			Aa = self.activationFunction(Za)
			Am = self.activationFunction(Zm)
			Z = pt.cat([Za, Zm], dim=1)
			A = pt.cat([Aa, Am], dim=1)
			output = A
			if(pt.isnan(A).any()):
				print("pt.isnan(A).any()")
				ex
		else:
			A = x @ self.W
			if(self.useBias):
				A = A + self.B
			output = self.activationFunction(A)
		return output

	def clipActivation(self, A):
		A = pt.clip(A, -activationMaxVal, activationMaxVal)	
		return A

	def multiplicativeEmulationFunctionPre(self, AprevLayer):
		AprevLayer = AprevLayer + multiplicativeEmulationFunctionOffsetVal
		AprevLayer = pt.clip(AprevLayer, multiplicativeEmulationFunctionPreMinVal, multiplicativeEmulationFunctionPreMaxVal)	
		AprevLayerM = pt.log(AprevLayer)
		return AprevLayerM
		
	def multiplicativeEmulationFunctionPost(self, ZmIntermediary):
		ZmIntermediary = pt.clip(ZmIntermediary, -multiplicativeEmulationFunctionPostMaxVal, multiplicativeEmulationFunctionPostMaxVal)
		Zm = pt.exp(ZmIntermediary)
		Zm = Zm - multiplicativeEmulationFunctionOffsetVal
		return Zm

