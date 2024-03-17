# ATORpt

### Author

Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

### Description:

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementations including 
receptive field feature/poly detection, parallel processed geometric hashing, end-to-end neural model. 
Classification of normalised snapshots (transformed patches) via ViT 

ATORpt contains various hardware accelerated implementations of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
	- supports classification of 2D image snapshots recreated from transformed mesh coordinates
		- perform independent, parallelised target prediction of object triangle data
- !useEndToEndNeuralModel (useStandardVIT)
	- useATORPTparallel:
		- uses parallel pytorch ATOR implementation
		- support corner/centroid features of the ATOR specification using third party libraries
		- third party feature detectors currently include, for point features: Harris/ShiTomasi/etc, centroids: segment-anything
		- supports simultaneous transformation of approx 9000 30x30px patches (ATOR 2D0D tri polys) on 12GB GPU
			- approx 10 images with 900 30x30px 2D0D tri polys per image, generated from approx 500 features per 500x500px image
			- approx 100x faster than useATORCPPserial
		- requires pytorch3d library
	- useATORCPPserial:
		- uses ATOR C++ executable to generate transformed patches (normalised snapshots)
		- requires all ATOR C++ prerequisites 
- useEndToEndNeuralModel (!useStandardVIT):
	- contains various modules for an end-to-end neural model of ATOR
	- designed to perform transformations of all image pixel coordinates in parallel
	- architecture layout provided (implementation incomplete)
	- supports feature detection via a CNN
	- currently supports 2DOD (2D/image input object data)
	- currently uses MNIST dataset to test affine(/euclidean/projective) invariance
	- also supports useMultKeys - modify transformer to support geometric hashing operations - experimental

See ATOR specification: https://www.wipo.int/patentscope/search/en/WO2011088497

Future:
Requires upgrading to support 3DOD (3D input object data)

#### Description (ATOR RF):

ATORpt RF is a receptive field implementation for ATOR feature/poly detection (ellipse centroids and tri corners)

ATOR RF currently contains its own unique implementation stack, although RF feature detection can be merged into the main code base.

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 
There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available
Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)

### License

MIT License

### Installation
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install tqdm
pip install transformers
pip install click
pip install opencv-python opencv-contrib-python
pip install kornia
pip install matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Execution
```
source activate pytorch3d
python3 ATORpt_main.py

source activate pytorch3d
python3 ATORpt_RFmain.py
```
