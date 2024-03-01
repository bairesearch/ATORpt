# ATORpt

### Author

Richard Bruce Baxter - Copyright (c) 2021-2024 Baxter AI (baxterai.com)

### Description

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementation based on an end-to-end parallel processed neural model 

ATORpt is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
- useATORCserialGeometricHashing:
	- uses ATOR C++ executable to generate transformed patches (normalised snapshots)
	- requires all ATOR C++ prerequisites 
- !useATORCserialGeometricHashing:
	- ATORpt contains various modules for an end-to-end neural model of ATOR
	- ATORpt is designed to perform transformations of all image pixel coordinates in parallel
	- architecture layout provided (implementation incomplete)
	- supports classification of 2D image snapshots recreated from transformed mesh coordinates - standard
		- perform independent, parallelised target prediction of object triangle data
	- supports feature detection via a CNN
	- currently supports 2DOD (2D/image input object data)
	- currently uses MNIST dataset to test affine(/euclidean/projective) invariance
	- also supports useMultKeys - modify transformer to support geometric hashing operations - experimental

### License

MIT License

### Installation
```
useATORCserialGeometricHashing:
	install all ATOR C++ prerequisites
	conda create --name pytorchsenv2 python=3.8
	source activate pytorchsenv2
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip3 install tqdm
	pip3 install transformers
!useATORCserialGeometricHashing:
	conda create -n pytorchsenv
	source activate pytorchsenv
	conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
	conda install tqdm
	conda install transformers
	conda install matplotlib
	pip install opencv-python opencv-contrib-python
	pip install kornia
```

### Execution
```
source activate pytorchenv
python3 ATORpt_main.py
```
