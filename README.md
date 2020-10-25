# Person ReID using Triplet Loss in Tensorflow
Tensorflow implementation of person re-identification using MobileNetV2 with variants of triplet loss on Market-1501 dataset. 
This project is part of the course offered by [July.edu](http://www.julyedu.com/) and thus I can't release the code in Tensorflow to the public, but I would love present and discuss about my code in a private way. Furthermore, I will reproduce the code in Pytorch and share it soon.
# Background
## Person Re-Identification
Person re-identification is the task of associating images of the same person taken from different cameras or from the same camera in different occasions.

Applications:
- surveillance and security - e.g. track the suspect of a crime scene, using multiple cameras.

Challenges:
- Illumination changes: changes of environments including day light and shade cause variances in the lightness.
- Resolution: most cameras are with low resolution - unable to trace with faces.
- Clothing: people may wear uniforms in schools or factories and therefore appearance-based algorithms cannot distinguish subjects.
- Occasion: backgrounds of walking people vary.
## Market-1501 Dataset
The Market-1501 dataset which was collected in front of a supermarket in Tsinghua University. A total of six cameras are used, including 5 high-resolution cameras, and one low-resolution camera. Overlap exists among different cameras. The original dataset contains 32,668 annotated bounding boxes of 1,501 identities, including 751 identities for training and 750 for testing. Each annotated identity is present in at least two cameras, so that cross-camera search can be performed. 

<div align=center><img src="https://github.com/Mu-ING/Person-ReID-Triplet-Loss-Model-based-on-Tensorflow/blob/master/Photos/Market1501.jpg" width="650" height="250"></div> 

## [SOTA of Person Re-Identification on Market-1501 from Paperwithcode](https://paperswithcode.com/sota/person-re-identification-on-market-1501) 

<div align=center><img src="https://github.com/Mu-ING/Person-ReID-Triplet-Loss-Model-based-on-Tensorflow/blob/master/Photos/SOTA%20ReID%20on%20Market-1501.png" width="1200" height="380"></div> 

## MobileNetV2
MobileNetV2 is a light but very effective feature extractor for object detection and segmentation. It builds upon the kernel of MobileNetV1 - depthwise separable convolution, and introduces two new features to the architecture: linear bottlenecks and inverted residuals (shortcut connections between the bottlenecks). The basic structure is shown below.

<div align=center><img src="https://github.com/Mu-ING/Person-ReID-Triplet-Loss-Model-based-on-Tensorflow/blob/master/Photos/MobileNetV2%20Architecture.jpg" width="350" height="250"></div> 

## Tricks
- Data augmentation
- Learning rate decay
- Label smoothing
- Transfer learning

## Triplet Loss
### 1.Basic Triplet Loss
Trhiplet loss was firstly introduced in the FaceNet paper, which is a loss function that trains a neural network to closely embed features of the same class (an anchor and a positive sample, both of which have the same identity) while maximizing the distance between feature embeddings of different classes (the anchor and a negative sample of a different identity). In this project, the anchor is one image of a person, the positive sample is one of the other images of the same person and the negative sample is an image of a different person. In basic triplet loss mode, the set of anchor, positive and negative is chosen randomly.

<div align=center><img src="https://github.com/Mu-ING/Person-ReID-Triplet-Loss-Model-based-on-Tensorflow/blob/master/Photos/tripletloss.jpg" width="580" height="150"></div>

### 2.Triplet Hard Loss

### 3.Triplet Semi-hard Loss

# Coding
## Environment
## Installation
## Dataset preparation
## Training & Validation
## Test

# Future Work
- Quadruplet Loss
- st-ReID

# References
- Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832), CVPR 2015
- Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian. [Scalable Person Re-identification: A Benchmark](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), ICCV 2015
- Weihua Chen, Xiaotang Chen, Jianguo Zhang, and Kaiqi Huang. [Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/abs/1704.01719), CVPR 2017
- Alexander Hermans, Lucas Beyer, and Bastian Leibe. [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737), 2017
- Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), CVPR 2018
- Guangcong Wang, Jianhuang Lai, Peigen Huang, and Xiaohua Xie.[Spatial-Temporal Person Re-identification](https://arxiv.org/abs/1812.03282),CVPR 2019
- TensorFlow Addons Losses: [TripletSemiHardLoss](https://www.tensorflow.org/addons/tutorials/losses_triplet)
