# Geometry-induced Implicit Regularization in Deep ReLU Neural Networks

## Description
We call $f_\theta(X)\in{\mathbb R}^{n_L\times n}$ the prediction of a sample $X\in{\mathbb R}^{n_0\times n}$ made by a deep ReLU neural network of parameter $\theta$. For $X$ fixed, we study geometrical properties of the set {  $f_\theta(X)$  |  $\theta$ varies  }  $\subset$ $\mathbb{R}^{n_L\times n}$.

We find that for almost all $\theta$, in the neiborhood of $f_\theta(X)$, the above set is a smooth manifold of fixed dimension. The dimension depends on $\theta$. It is called **batch functional dimension**.

When computed on the learning sample and the test sample, the experiments show that:
-  The batch functional dimension decays during learning.
- When the network width increases, the batch functional dimension after learning first increases, then becomes stable.
- When corrupting the input learning sample with noise, the batch functional dimension for the optimized parameters increases with the level of noise.
- When completing the learning sample with samples having a random output, for  the optimized parameters of large networks (i.e. when the implicit regularization occurs), the batch functional dimension increases with the number of additional samples.

The batch functional dimension computed on a large Gaussian input sample $X$, called the **computable full functional dimension** often remains close to the the number of parameters, indicating that the parameters of the networks are locally identifiable. 

This also shows that the complexity of the network, as measured by the batch functional dimension, is smaller than the number of parameters on the support of the distribution of the input but remains large on the whole domain ${\mathbb R}^{n_0}$. This indicates that the geometry-induced implicit regularization only occurs on the domain of the distribution of input.

We call this phenomenon **geometry-induced implict regularization**.

The details of the theory and the experiments are in the article below.

## Reference

This repository contains the codes of the experiments in the article:

*[Geometry-induced Implicit Regularization in Deep ReLU Neural Networks](https://arxiv.org/abs/2402.08269)*.

When citing this work or results obtained with the codes, please cite

@article{BonaPellissier-Malgouyres-Bachoc,
  title={Geometry-induced Implicit Regularization in Deep ReLU Neural Networks},
  author={ Bona-Pellissier, Joachim and  Malgouyres, Fran{\c{c}}ois and Bachoc, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2402.08269},
  year={2024}
}

