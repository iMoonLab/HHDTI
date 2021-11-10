import os
import numpy as np
# import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import random
from hypergraph_utils import _generate_G_from_H, generate_G_from_H
# import matplotlib.pyplot as plt
from scipy import stats

random.seed(3)
torch.manual_seed(3)

def pre_processed_DTInet():

    dataset_dir = os.path.sep.join(['DTInet'])
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_protein_remove_homo.txt']), dtype=np.int32)
    # print(len(i_m), len(i_m[0]))
    # a = np.genfromtxt(os.path.sep.join([dataset_dir, 'drug_vector_d100.txt']), dtype=np.float)
    # b = np.genfromtxt(os.path.sep.join([dataset_dir, 'protein_vector_d400.txt']), dtype=np.float)
    # print(a)
    # print(len(a), len(a[0]))  # 708, 100
    # print(len(b), len(b[0]))  #  1512, 400
    edge = []
    for i in range(len(i_m)):
        for j in range(len(i_m[0])):
            if i_m[i][j] == 1:
                edge.append([i, j])
    print(len(edge))

    with open(os.path.sep.join([dataset_dir, "drug_target_interaction_remove_homo.txt"]), "w") as f0:
        for i in range(len(edge)):
            s = str(edge[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)


# pre_processed_DTInet()


def load_data_DTInet(dataset_train="DTInet_train_0.1_0", dataset_test="DTInet_test_0.1_0"):  # 更改测试集
    dataset_dir = os.path.sep.join(['DTInet'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("DTInet_all")]), dtype=np.int32)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_protein.txt']), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1

    np.set_printoptions(threshold=np.inf)

    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()

    drug_feat = torch.eye(708)
    protein_feat = torch.eye(1512)

    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_drug_disease.txt']), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'mat_protein_disease.txt']), dtype=np.int32))

    # print(drugDisease.size())  # 708, 5603
    # print(proteinDisease.size())  # 1512, 5603
    print("DTInet", H.size())  # 1512, 708

    return drugDisease, proteinDisease, drug_feat, protein_feat, H, H_T, edge_test, test


# load_data_DTInet()


def generate_data_2(dataset_str="drug_target_interaction_remove_homo"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['DTInet'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    print(edge)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)

    drugs = []
    targets = []
    for i in edge_shuffled:
        if i[0] not in drugs:
            drugs.append(i[0])
        if i[1] not in targets:
            targets.append(i[1])

    test_ration = [0.1]
    for d in test_ration:
        for a in (range(10)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) < len(edge_test):
                x1 = random.sample(range(0, 708), 1)[0]
                y1 = random.sample(range(0, 1512), 1)[0]
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros:
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "DTInet_train_{ratio}_{fold}_remove_homo.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "DTInet_test_{ratio}_{fold}_remove_homo.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    # with open(os.path.sep.join([dataset_dir,  "DTInet_all.txt"]), "w") as f3:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f3.write(s)


# generate_data_2()




