# deeplearning_collection
## architecture improvement
- a-softmax
  - [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063), CVPR 2017
- Additive Angular Margin
  - [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843), ICLR 2017
- Max over time pooling
  - [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), EMNLP 2014
- Self attention
  - [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130), ICLR 2017
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Transformer, noam learning rate warnup, NIPS 2017
  - [DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/abs/1709.04696), AAAI 2018
- Stack-augmented Parser-Interpreter Neural Network (SPINN)
  - [A Fast Unified Model for Parsing and Sentence Understanding](https://arxiv.org/abs/1603.06021), ACL 2016
- [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576), ICLR 2017

## training technic
- Optimization methods
  - [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417), powersign, addsign
  - [SGDR: stochastic gradient descent with restarts](https://arxiv.org/abs/1608.03983), restart decay, ICLR 2017
- Virtual adversarial training
  - [Distributional Smoothing with Virtual Adversarial Training](https://arxiv.org/abs/1507.00677), ICLR 2016
  - [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725), ICLR 2017
  - [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](https://arxiv.org/abs/1704.03976), ICLR 2016
- Curriculum learning
  - [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf), ICML 2009
  - [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003), ICML 2017
- Label smoothing
  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), spread the 1 − β probability mass uniformly over all classes, CVPR 2016
  - [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548),  confidence penalty, label smoothing, distributes the remaining probability mass proportionally to the marginal probability of classes, ICLR 2017
  - [Towards better decoding and language model integration in sequence to sequence models](https://arxiv.org/abs/1612.02695), neighborhood smoothing scheme, Google Brain 2016
  - [Improved training for online end-to-end speech recognition systems](https://arxiv.org/abs/1711.02212), add a regularization term to the CTC objective function which consists of the KL divergence between the network’s predicted distribution P and a uniform distribution U over labels, ICASSP 2018

## data augmentation
- Adversarial domain adaptation
  - [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464), CVPR 2017
  - [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213)
  - [Adversarial Feature Augmentation for Unsupervised Domain Adaptation](https://arxiv.org/abs/1711.08561)
  - [Addressing Appearance Change in Outdoor Robotics with Adversarial Domain Adaptation](https://arxiv.org/abs/1703.01461), IROS 2017
  - [Incremental Adversarial Domain Adaptation for Continually Changing Environments](https://arxiv.org/abs/1712.07436), ICRA 2018

## gan
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), ICCV 2017
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), NIPS 2017
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
- [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848), NIPS 2017
- [Be Your Own Prada: Fashion Synthesis with Structural Coherence](https://arxiv.org/abs/1710.07346), ICCV 2017

## style transfer
- [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086), NIPS 2017

## super resolution
- [Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform](https://github.com/xinntao/CVPR18-SFTGAN), CVPR 2018

## system
- [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), NIPS 2012
- [ring allreduce](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)
- [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799)

## dataset
- Kaggle
  - [plant-seedlings-classification](https://www.kaggle.com/c/plant-seedlings-classification/data)
- [iNaturalist 2018 Competition](https://github.com/visipedia/inat_comp)
- [Chinese Text in the Wild](https://arxiv.org/abs/1803.00085v1)
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition](https://arxiv.org/abs/1607.08221)
- [nlp-datasets](https://github.com/niderhoff/nlp-datasets/blob/master/README.md)
- [xmedia](http://www.icst.pku.edu.cn/mipl/xmedia/)
- [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## tools
- [Lucid](https://github.com/tensorflow/lucid)
