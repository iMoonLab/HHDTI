# Exploring Complex and Heterogeneous Correlations on Hypergraph for the Prediction of Drug-Target Interactions

HHDTI is a deep learning method that uses hypergraphs to model heterogeneous biological networks, integrates various interaction information, and accurately captures the topological properties of individual in the biological network to generate suitable structured embeddings for interactions prediction.

# Quick start
We provide an example script to run experiments on deepDTnet_20 dataset:

+   Run `python train.py`: predict drug-target interactions, and evaluate the results with 5-fold cross-validation.
+   You can change the dataset used, adjust the learning rate, hidden layer dimensions, etc
    `python train.py --dataset DTInet --lr 0.002 --hidden 64`


# Code

+   `train.py`:run HHDTI to predict drug-target interactions
+   `models.py`:the modules of VHAE and HGNN
+   `layers.py`:the layers used in VHAE and HGNN
+   `kl_loss.py`:KL divergence loss function for training
+   `hypergraph_utils.py`:generate degree matrices of hypergraph
+   `utils_deepDTnet.py`:load deepDTnet dataset for training and testing
+   `utils_DTInet.py`:load DTInet dataset for training and testing
+   `utils_KEGG_MED.py`:load KEGG_MED dataset for training and testing


# Data
+   `DTInet`:The DTIs in the DTINet_17 dataset used for training and testing have been divided into the form of 10-fold cross-validation, as well as disease-drug associations, disease-target associations. Dataset source: https://github.com/luoyunan/DTINet
+   `deepDTnet`:The DTIs in the deepDTnet_20 dataset used for training and testing have been divided into the form of 10-fold cross-validation, as well as disease-drug associations, disease-target associations. Dataset source: https://github.com/ChengF-Lab/deepDTnet
+   `KEGG_MED`:The DTIs in the KEGG_MED dataset used for training and testing have been divided into the form of 10-fold cross-validation, as well as disease-drug associations, disease-target associations(the disease-drug associations matrix file is larger than 25M, and cannot be uploaded temporarily). Dataset source: http://drugtargets.insight-centre.org/
# Note
+   You can also use your own dataset for experiments. Due to the different representations of datasets, it is not recommended that you use our code to load data.The input of HHDTI is 4 incidence matrices, which respectively represent drug-target interactions, target-drug interactions, disease-drug associations, disease-target associations, among which the drug-target interactions incidence matrix and the target-drug interaction incidence matrix are transposed to each other.

# Requirements
+   python(v3.7.0)
+   numpy(v1.18.1)
+   pytorch(v1.1.0)
+   scikit-learn(v0.22.1)

# License
Our code is released under MIT License (see LICENSE file for details).
