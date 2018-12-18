# deeplearning_collection
## comparison
- [Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/abs/1806.06259), a nice and extensive comparison between ELMo, InferSent, Google Universal Sentence Encoder, p-mean, Skip-thought
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
- [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/abs/1512.05193v4), siamese architecture TACL 2016
- [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726), Using word vector learning as inspiration, we
propose an objective function that abstracts the skip-gram model to the sentence level. NIPS 2015
- [ELMo, Embeddings from Language Models](https://arxiv.org/abs/1802.05365), word vectors are learned functions of the internal states of a deep bidirectional language model, NAACL 2018
- [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146), ACL 2018
- [Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information](https://arxiv.org/abs/1805.11360)
- [A Hybrid Learning Scheme for Chinese Word Embedding](http://www.aclweb.org/anthology/W18-3011), word, character and component, ACL 2018
- [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603), ICLR 2017
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), a bidirectional Transformer, representations are jointly conditioned on both left and right context in all layers
- [Matrix Power Normalized Covariance Pooling For Deep Convolutional Networks](http://peihuali.org/iSQRT-COV/index.html), Matrix Power Normalized Covariance pooling, (Fast-)MPN-COV, ICCV 2017, CVPR 2018

## training technic
- Initialization scheme
  - [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks](https://arxiv.org/abs/1806.05393), [Delta-Orthogonal Initialization](https://www.tensorflow.org/api_docs/python/tf/contrib/framework/convolutional_delta_orthogonal)
- Optimization methods
  - [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235), Adafactor
  - [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/abs/1709.07417), powersign, addsign
  - [SGDR: stochastic gradient descent with restarts](https://arxiv.org/abs/1608.03983), restart decay, ICLR 2017
  - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820), [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy)
  - [SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE](https://arxiv.org/abs/1704.00109), Cyclic Cosine Annealing, before restart take a snapshot
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
 - [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813), JAIR Volume 16, pages 321-357, 2002
 - [ADASYN: Adaptive synthetic sampling approach for imbalanced learning](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf), IEEE 2008

## gan
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), ICCV 2017
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), NIPS 2017
- [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
- [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848), NIPS 2017
- [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732), ECCV 2018
- [Be Your Own Prada: Fashion Synthesis with Structural Coherence](https://arxiv.org/abs/1710.07346), ICCV 2017
- [Learning Face Age Progression: A Pyramid Architecture of GANs](https://arxiv.org/abs/1711.10352), CVPR 2018
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)

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
- [Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)
- [OpenImages](https://storage.googleapis.com/openimages/web/index.html), Containing 15.4M bounding-boxes for 600 categories on 1.9M images
- [YouTube8M](https://research.google.com/youtube8m/)
- [SQuAD 2.0 The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

## tools
- [Lucid](https://github.com/tensorflow/lucid)
- [seedbank](http://tools.google.com/seedbank/)
- [Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)
- [HanLP: Han Language Processing](https://github.com/hankcs/HanLP)
- [What-If Tool](https://pair-code.github.io/what-if-tool/)

## rules
- [rules-of-ml](https://developers.google.cn/machine-learning/rules-of-ml/)
- [Practical Advice for Building Deep Neural Networks](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/)

