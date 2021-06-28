# Pretrained EfficientDet Network For Object Detection

This repository provides a pretrained EfficientDet-D0[1] object detection network for MATLAB&reg;. 

Requirements
------------  

- MATLAB&reg; R2021a or later
- Deep Learning Toolbox&trade;
- Computer Vision Toolbox&trade;
- Deep Learning Toolbox Converter for ONNX Model Format&trade; Support Package

Overview
--------

This repository provides the EfficientDet-D0 network trained to detect different object categories including person, car, traffic light, etc. This pretrained model is trained using COCO 2017 [3] dataset which have 80 different object categories.

EfficientDet-D0 largely follows the one stage object detector paradigm and uses pre-defined anchors to detect objects. With a weighted bi-directional feature pyramidal network enhanced with fast normalization, it leverages easy and fast multi-scale feature fusion from different levels of the backbone network.  


Getting Started
---------------

Download or clone this repository to your machine and open it in MATLAB&reg;.

### Setup
Add path to the source directory.

```
addpath('src');
```

### Download the pretrained network
Use the below helper to download the pretrained network.

```
model = helper.downloadPretrainedEfficientDetD0;
net = model.net;
```

Detect Objects Using Pretrained EfficientDet-D0
-----------------------------------------------

```
% Read test image.
img = imread('visionteam.jpg');

% Get classnames for COCO dataset.
classNames = helper.getCOCOClasess;

% Perform detection using pretrained model.
executionEnvironment = 'auto';
[bboxes,scores,labels] = detectEfficientDetD0(net, img, classNames, executionEnvironment);

% Visualize results.
annotations = string(labels) + ": " + string(round(100*scores)) + "%";
img = insertObjectAnnotation(img, 'rectangle', bboxes, cellstr(annotations));
figure, imshow(img);
```
![alt text](images/result.png?raw=true)


Train Custom EfficientDet-D0 Using Transfer Learning
----------------------------------------------------
Transfer learning enables you to adapt a pretrained EfficientDet-D0 network to your dataset. Create a custom EfficientDet-D0 network for transfer learning with a new set of classes and train using the `efficientDetD0TransferLearn.m` script.

Code Generation for EfficientDet-D0
-----------------------------------
Code generation enables you to generate code and deploy EfficientDet-D0 on multiple embedded platforms.

Run `codegenEfficientDetD0.m`. This script calls the `efficientDetD0_predict.m` entry point function and generate CUDA code for it. It will run the generated MEX and gives output.

| Model | Inference Speed (FPS) | 
| ------ | ------ | 
| EfficientDet-D0 w/o codegen | 4.8437 |
| EfficientDet-D0 with codegen | 27.3658 |

- Performance (in FPS) is measured on a TITAN-RTX GPU using 512x512 image.

For more information about codegen, see [Deep Learning with GPU Coder](https://www.mathworks.com/help/gpucoder/gpucoder-deep-learning.html)

Accuracy
---------

| Model | Input image resolution | mAP  | Size (MB) | Classes |
| ------ | ------ | ------ | ------ | ------ |
| EfficientDet-D0 | 512 x 512 | 33.7 | 15.9 | [coco class names](src/+helper/coco-classes.txt) |

- mAP for models trained on the COCO dataset is computed as average over IoU of .5:.95.

EfficientDet-D0 Network Details
--------------------------------

EfficientDets are a family of object detection models. These are developed based on the advanced EfficientNet backbones, a new BiFPN module, and compound scaling technique. They follow the one-stage detectors paradigm.

![alt text](images/network.png?raw=true)

- **Backbone:** EfficientNets[2] are used as backbone networks for this class of object detectors. EfficientNets are a class of convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. It has eight variants out of which, EfficientNet-B0 is used as the backbone for this model.

- **BiFPN Module:** A weighted bi-directional feature pyramidal network enhanced with fast normalization, which enables easy and fast multi-scale feature fusion. This module takes level 3-7 features (P3, P4, P5, P6, P7) from the backbone network and repeatedly applies top-down and bottom-up bidirectional feature fusion. These fused features are fed to a class prediction network and box prediction network to produce object class and bounding box predictions respectively. The class and box prediction network weights are shared across all levels of features. 

  This module is implemented using a combination of layers such as convolution, resize, element-wise multiplication, element-wise addition, element-wise division etc.

- **Scaling:** A compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. This method helps in optimizing both accuracy and efficiency for the model.

- **Class Prediction Net:** This network processes the fused features from the previous BiFPN modules and produces class prediction outputs. This network is implemented using a combination of layers such as convolution, sigmoid, element-wise multiplication etc. 

- **Box Prediction Net:** This network processes the fused features from the previous BiFPN modules and produces bounding box prediction outputs. This network is implemented using a combination of layers such as convolution, sigmoid, element-wise multiplication etc.


References
-----------

[1] Tan, Mingxing, Ruoming Pang, and Quoc V. Le. "Efficientdet: Scalable and efficient object detection." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10781-10790. 2020.

[2] Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International Conference on Machine Learning. PMLR, 2019.

[3] Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).

Copyright 2021 The MathWorks, Inc.
