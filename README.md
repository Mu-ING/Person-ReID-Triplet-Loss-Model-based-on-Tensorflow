# Person-ReID-Triplet-Loss-Model-based-on-Tensorflow
Tensorflow implementation of person re-identification using MobileNetV2 with variants of triplet loss on Market1501 dataset
# Background of Person Re-Identification

# MobileNetV2
MobileNetV2 is a light but very effective feature extractor for object detection and segmentation. It builds upon the kernel of MobileNetV1 - depthwise separable convolution, and introduces two new features to the architecture: linear bottlenecks and inverted residuals (shortcut connections between the bottlenecks). The basic structure is shown below.

<div align=center><img src="https://github.com/Mu-ING/Person-ReID-Triplet-Loss-Model-based-on-Tensorflow/blob/master/Photos/MobileNetV2%20Architecture.jpg" width="350" height="250"></div> 

# Triplet Loss
## 1.Basic
## 2.Triplet Hard Loss

# References
- Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832), CVPR 2015  
- Alexander Hermans, Lucas Beyer, and Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737), 2017
- Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen, [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), CVPR 2018
