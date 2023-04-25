# ATORpt

### Author

Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

### Description

Axis Transformation Object Recognition (ATOR) for PyTorch - experimental implementation based on an end-to-end parallel processed neural model 

ATORpt is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for PyTorch

- ATORpt contains various modules for an end-to-end neural model of ATOR
- ATORpt is designed to perform transformations of all image pixel coordinates in parallel
- architecture layout provided (implementation incomplete)
- supports classification of transformed mesh coordinates with a vision transformer (vit) - experimental
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
conda create -n pytorchenv
source activate pytorchenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tqdm
conda install matplotlib
```

### Execution
```
source activate pytorchenv
python3 ATORpt_main.py
```
