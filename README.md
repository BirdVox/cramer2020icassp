# Chirping up the right tree: incorporating biological taxonomies into deep bioacoustic classifiers

This repository contains the code used for the experiments in our paper:

[Chirping up the Right Tree: Incorporating Biological Taxonomies into Deep Bioacoustic Classifiers](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_taxonet_icassp_2020.pdf)  
J. Cramer, V. Lostanlen, A. Farnsworth, J. Salamon, J.P. Bello  
In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, May 2020.

You can watch our ICASSP 2020 presentation [here](https://youtu.be/A16ObKJ6lLY), or view the slides [here](icassp2020-slides.pdf).

#### Abstract
Class imbalance in the training data hinders the generalization ability of machine listening systems.
In the context of bioacoustics, this issue may be circumvented by aggregating species labels into super-groups of higher taxonomic rank: genus, family, order, and so forth.
However, different applications of machine listening to wildlife monitoring may require different levels of granularity.
This paper introduces TaxoNet, a deep neural network for structured classification of signals from living organisms.
TaxoNet is trained as a multitask and multilabel model, following a new architectural principle in end-to-end learning named hierarchical composition: shallow layers extract a shared representation to predict a root taxon, while deeper layers specialize recursively to lower-rank taxa.
In this way, TaxoNet is capable of handling taxonomic uncertainty, out-of-vocabulary labels, and open-set deployment settings.
An experimental benchmark on two new bioacoustic datasets (ANAFCC and BirdVox-14SD) leads to state-of-the-art results in bird species classification.
Furthermore, on a task of coarse-grained classification, TaxoNet also outperforms a flat single-task model trained on aggregate labels.


#### Setup

To get started with our code:
* Clone the repository: `git clone https://github.com/BirdVox/cramer2020icassp.git`
* Enter the repository directory: `cd cramer2020icassp`
* Install dependencies: `pip install -r requirements.txt`
* Edit paths in `localmodule.py` to accomodate your local workspace

Note that this code requires Python 3.6+.

#### Obtaining Data:
* ANAFCC (training and validation): [https://doi.org/10.5281/zenodo.3666782](https://doi.org/10.5281/zenodo.3666782)
* BirdVox-DCASE-20k (noise for data augmentation): [https://archive.org/details/BirdVox-DCASE-20k](https://archive.org/details/BirdVox-DCASE-20k)
* BirdVox-14SD (testing): [https://doi.org/10.5281/zenodo.3667094](https://doi.org/10.5281/zenodo.3667094)

#### Running Code
Relevant scripts have filenames starting with a number, indicating the order in which to run them. After editing `localmodule.py` to reflect your workspace, scripts can simply be run via `python <script_path>`.
