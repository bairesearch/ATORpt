# ATORpt

### Author

Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

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
		- support3DOD:generate3DODfrom2DOD uses intel-isl MiDaS library (3D input object data)
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
Requires upgrading to support3DOD:generate3DODfromParallax

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
pip install git+https://github.com/facebookresearch/segment-anything.git (required for useATORPTparallel:useFeatureDetectionCentroids and ATORpt_RFdetectEllipsesSA)
pip install timm (required for useATORPTparallel:generate3DODfrom2DOD only)
pip install lovely-tensors
```

### Execution
```
source activate pytorch3d
python3 ATORpt_main.py

source activate pytorch3d
python3 ATORpt_RFmain.py
```

### Acknowledgements

#### PyTorch3D (useATORPTparallel)

* [Ravi, N., Reizenstein, J., Novotny, D., Gordon, T., Lo, W. Y., Johnson, J., & Gkioxari, G. (2020). Accelerating 3d deep learning with pytorch3d. arXiv preprint arXiv:2007.08501.](https://arxiv.org/abs/2007.08501)

#### Segment Anything (useATORPTparallel:useFeatureDetectionCentroids)

* [Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015-4026).](https://arxiv.org/abs/2304.02643)

#### MiDaS (useATORPTparallel:generate3DODfrom2DOD)

* [Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3), 1623-1637.](https://arxiv.org/abs/1907.01341)
* [Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 12179-12188).](https://arxiv.org/abs/2103.13413)


------------------------------------------------------------------------

# ATORpt_RF

### Author

Richard Bruce Baxter - Copyright (c) 2021-2025 Baxter AI (baxterai.com)

### Description:

* ATORpt_RFmainFT: Perform ATOR receptive field (RF) ellipse detection using pytorch RF filters/masks (FT) (hardware accelerated).
* ATORpt_RFmainSA: Perform ATOR receptive field (RF) ellipse detection using segment-anything (SA) library (hardware accelerated) rather than RF filters.
* ATORpt_RFmainCV: Perform ATOR receptive field (RF) ellipse detection using open-cv (CV) library (non-hardware accelerated) rather than RF filters.

ATORpt RF is a receptive field implementation for ATOR feature/poly detection (ellipse centroids and tri corners).

ATOR RF currently contains its own unique implementation stack, although RF feature detection can be merged into the main code base.

ATORpt RF supports ellipsoid features (for centroid detection), and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)
* features can still be detected where there are no point features available

Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space).

### License:

MIT License

### Installation:

Same as ATORpt

### Execution
```
set RFmethod = "FT"
source activate pytorch3d
python ATORpt_RFmainFT.py images/leaf1.png

set RFmethod = "SA"
source activate pytorch3d
python ATORpt_RFmainSA.py images/leaf1.png

set RFmethod = "CV"
source activate pytorch3d
python ATORpt_RFmainCV.py images/leaf1.png
```

### Acknowledgements

#### Segment Anything (ATORpt_RFdetectEllipsesSA)

* [Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015-4026).](https://arxiv.org/abs/2304.02643)





