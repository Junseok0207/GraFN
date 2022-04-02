# GraFN: Semi-Supervised Graph Node Classification with Few Labels via Non-Parametric Distribution Assignment

<p align="center">
  <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&logo=Python&logoColor=white"/>
  <a href="https://pytorch.org/" alt="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
  <a href="https://sigir.org/sigir2022/" alt="Conference">
    <img src="https://img.shields.io/badge/SIGIR'22-lightgray"/></a>


The official source code for "GraFN: Semi-Supervised Graph Node Classification with Few Labels via Non-Parametric Distribution Assignment", accepted at SIGIR 2022(Short Paper).  

## Overview
Despite the success of Graph Neural Networks (GNNs) on various applications, GNNs encounter significant performance degradation when the amount of supervision signals, i.e., number of labeled
nodes, is limited, which is expected as GNNs are trained solely based on the supervision obtained from the labeled nodes. On the other hand, recent self-supervised learning paradigm aims to train
GNNs by solving pretext tasks that do not require any labeled nodes, and it has shown to even outperform GNNs trained with few labeled nodes. However, a major drawback of self-supervised
methods is that they fall short of learning class discriminative node representations since no labeled information is utilized during training.
To this end, we propose a novel semi-supervised method for graphs, GraFN, that leverages few labeled nodes to ensure nodes that belong to the same class to be grouped together, thereby achieving
the best of both worlds of semi-supervised and self-supervised methods. Specifically, GraFN randomly samples support nodes from labeled nodes and anchor nodes from the entire graph. Then, it
minimizes the difference between two predicted class distributions that are non-parametrically assigned by anchor-supports similarity from two differently augmented graphs. We experimentally show
that GraFN surpasses both the semi-supervised and self-supervised methods in terms of node classification on real-world graphs.

<img width="80%" src="Img/Architecture.pdf"></img>