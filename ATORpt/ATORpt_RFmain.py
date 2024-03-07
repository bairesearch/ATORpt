"""ATORpt_RFmain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORpt_main.py

# Usage:
source activate pytorchsenv2
python ATORpt_RFmain.py images/leaf1.png

# Description:
ATORpt RF is a receptive field implementation for ATOR feature detection (ellipse centroids and tri corners)

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 
There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available
Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORpt will also support point (corner/centroid) features of the ATOR specification using a third party library; 
https://www.wipo.int/patentscope/search/en/WO2011088497

# Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)

"""

import os
import torch as pt
import numpy as np
import click
import cv2
import copy
import torch as pt
import torch.nn.functional as F

from ATORpt_RFglobalDefs import *
#import ATORpt_RFdetectEllipses
import ATORpt_RFgenerate
import ATORpt_RFapply

@click.command()
@click.argument('inputimagefilename')
			
def main(inputimagefilename):
	#ATORpt_RFdetectEllipses.main(inputimagefilename)
	RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers = ATORpt_RFgenerate.prepareRFhierarchyAccelerated()
	ATORpt_RFapply.updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename)	#trial image

if __name__ == "__main__":
	main()
