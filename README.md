# Attention-Based deep Multiple Instance Learning with within-bag sampling
<a href="mailto:nadezhda.koriakina@it.uu.se">Nadezhda Koriakina</a>:envelope:, <a href="mailto:natasa.sladoje@it.uu.se">Nataša Sladoje</a> and <a href="mailto:joakim.lindblad@it.uu.se">Joakim Lindblad</a>

Pytorch implementation of Attention-Based deep Multiple Instance Learning (ABMIL) [[1]](#1) with within-bag sampling according to [ISPA 2021](https://www.isispa.org/) paper ["The Effect of Within-Bag Sampling on End-to-End Multiple Instance Learning"](#2). 

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [How to use](#how-to-use)
4. [Citation](#citation)
5. [References](#references)
6. [Acknowledgements](#acknowledgements)

## Overview
- Code for creating QMNIST-bags and Imagenette-bags datasets
- Code for training and evaluating ABMIL with/without within bag sampling for QMNIST-bags and Imagenette-bags

Additions to the [original implementation](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) of ABMIL:
- Within-bag sampling option
- Options of GPU training with one or three GPUs
- Validation technique with moving average
- Evaluation by computing Area Under the Receiver Operating Characteristic (AUC) curve at a bag and instance level

## Dependencies
Python version 3.6.13. Main dependencies: reqs.txt

## How to use
- `Create_QMNIST_bags_dataset.ipynb`: code for creating QMNIST-bags dataset. There are options to choose number of bags, number of instances in a bag, percent of key instances in a positive bag. The code is made to create bags without repeating instances in a bag
- `Create_IMAGENETTE_bags_dataset.ipynb`: code for creating Imagenette-bags dataset. There are options to choose number of bags, number of instances in a bag, percent of key instances in a positive bag, augmentation of images
- `MAIN_ABMIL_with_within_bag_sampling_QMNIST.ipynb` and `MAIN_ABMIL_with_within_bag_sampling_IMAGENETTE.ipynb`: code for training and evaluating ABMIL with/without within bag sampling for QMNIST-bags and Imagenette-bags datasets correspondingly.

Restart the kernel if changes to the internal codes attention_model.py, dataloaders.py and evaluation.py are made.

<ins>Note:</ins> the code is created for QMNIST and Imagenette data and might require changes if custom data is used.

## Citation

## References
<a id="1">[1]</a> 
M.  Ilse,  J.  Tomczak,  and  M.  Welling,  “Attention-based  deep  multiple instance   learning,”  in International conference on machine learning.PMLR, 2018, pp. 2127–2136.
<a id="2">[2]</a> 
N. Koriakina, N. Sladoje and J. Lindblad, "The Effect of Within-Bag Sampling on End-to-End Multiple Instance Learning," 2021 12th International Symposium on Image and Signal Processing and Analysis (ISPA), 2021, pp. 183-188, doi: 10.1109/ISPA52656.2021.9552170.


## Acknowledgements
This work is supported by: VINNOVA MedTech4Health grant 2017-02447 and Swedish Research Council grants 2015-05878 and 2017-04385. A part of computations was enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE), partially funded by the Swedish Research Council through grant no. 2018-05973.

