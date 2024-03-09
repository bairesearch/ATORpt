# ATORpt

### Author

Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

### Description:

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementations including 
receptive field feature/poly detection, parallel processed geometric hashing, end-to-end neural model. 
Classification of normalised snapshots (transformed patches) via ViT 

ATORpt contains various hardware accelerated implementations of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
- !useEndToEndNeuralModel (useStandardVIT)
	useATORPTparallel:
		- uses third party feature detectors (point feature and segmenter: segment-anything)
		- uses parallel pytorch ATOR implementation
	useATORCPPserial:
		- uses ATOR C++ executable to generate transformed patches (normalised snapshots)
		- requires all ATOR C++ prerequisites 
- useEndToEndNeuralModel (!useStandardVIT):
	- ATORpt contains various modules for an end-to-end neural model of ATOR
	- ATORpt is designed to perform transformations of all image pixel coordinates in parallel
	- architecture layout provided (implementation incomplete)
	- supports classification of 2D image snapshots recreated from transformed mesh coordinates - standard
		- perform independent, parallelised target prediction of object triangle data
	- supports feature detection via a CNN
	- currently supports 2DOD (2D/image input object data)
	- currently uses MNIST dataset to test affine(/euclidean/projective) invariance
	- also supports useMultKeys - modify transformer to support geometric hashing operations - experimental

#### Description (ATOR RF):

ATORpt RF is a receptive field implementation for ATOR feature/poly detection (ellipse centroids and tri corners)

ATOR RF currently contains its own unique implementation stack, although RF feature detection can be merged into the main code base.

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 
There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available
Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORpt will also support point (corner/centroid) features of the ATOR specification using a third party library; 
https://www.wipo.int/patentscope/search/en/WO2011088497

Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)

### License

MIT License

### Installation
```
conda create --name pytorchsenv2 python=3.8
source activate pytorchsenv2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm
pip3 install transformers
pip3 install click
pip3 install opencv-python opencv-contrib-python
pip3 install kornia
pip3 install matplotlib
install all ATOR C++ prerequisites	(required for useATORCPPserial only)
pip3 install git+https://github.com/facebookresearch/segment-anything.git (required for useATORPTparallel only)
	pip3 install opencv-python pycocotools matplotlib onnxruntime onnx
	download default checkpoint (ViT_h SAM model) https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Execution
```
source activate pytorchsenv2
python3 ATORpt_main.py

source activate pytorchsenv2
python3 ATORpt_RFmain.py
```
