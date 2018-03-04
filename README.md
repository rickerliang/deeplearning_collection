# deeplearning_paper_collection
## architecture improvement
- a-softmax
  - [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
- [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)
- Max over time pooling
  - [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- Self attention
  - [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/abs/1709.04696)
- Stack-augmented Parser-Interpreter Neural Network (SPINN)
  - [A Fast Unified Model for Parsing and Sentence Understanding](https://arxiv.org/abs/1603.06021)
- [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)

## training technic
- Virtual adversarial training
  - [Distributional Smoothing with Virtual Adversarial Training](https://arxiv.org/abs/1507.00677)
  - [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)
  - [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](https://arxiv.org/abs/1704.03976)
- Curriculum learning
  - [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf), ICML, 2009
  - [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003), ICML, 2017
- Label smoothing
  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), spread the 1 − β probability mass uniformly over all classes, CVPR, 2016
  - [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548),  distributes the remaining probability mass proportionally to the marginal probability of classes, ICLR, 2017
  - [Towards better decoding and language model integration in sequence to sequence models](https://arxiv.org/abs/1612.02695), neighborhood smoothing scheme, Google Brain, 2016

## data augmentation
- adversarial domain adaptation
  - [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464)
  - [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213)
  - [Adversarial Feature Augmentation for Unsupervised Domain Adaptation](https://arxiv.org/abs/1711.08561)
  - [Addressing Appearance Change in Outdoor Robotics with Adversarial Domain Adaptation](https://arxiv.org/abs/1703.01461)

## gan
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

## style transfer
- [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086)

## system
- [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
- [ring allreduce](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)
