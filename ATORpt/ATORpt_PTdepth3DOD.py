"""ATORpt_PTdepth3DOD.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATORpt_main.py

# Usage:
See ATORpt_main.py

# Description:
ATORpt PT depth 3DOD

"""

import torch as pt
import matplotlib.pyplot as plt
import ATORpt_operations

from ATORpt_globalDefs import *

if(support3DOD):

	#https://pytorch.org/hub/intelisl_midas_v2/

	model_type = "DPT_Large"	 # MiDaS v3 - Large	 (highest accuracy, slowest inference speed)
	#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid	(medium accuracy, medium inference speed)
	#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

	midas = pt.hub.load("intel-isl/MiDaS", model_type)
	midas.to(device)
	midas.eval()

	midas_transforms = pt.hub.load("intel-isl/MiDaS", "transforms")
	if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
		transform = midas_transforms.dpt_transform
	else:
		transform = midas_transforms.small_transform

	def deriveImageDepth(image):
		input_batch = transform(image).to(device)
		with pt.no_grad():
			prediction = midas(input_batch)
			prediction = pt.nn.functional.interpolate(prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False).squeeze()
		if(debug3DODgeneration):
			imageDepth = prediction.cpu().numpy()
			plt.imshow(imageDepth)
			plt.show()
		prediction = prediction.to(devicePreprocessing)
		return prediction

