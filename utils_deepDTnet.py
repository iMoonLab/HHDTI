import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import random
import scipy.io
from sklearn.decomposition import non_negative_factorization

random.seed(3)
torch.manual_seed(4)

def pre_processed_DTnet_1():
    dataset_dir = os.path.sep.join(['deepDTnet'])
    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)
    print(len(i_m), len(i_m[0]))

    edge = []
    for i in range(len(i_m)):
        for j in range(len(i_m[0])):
            if i_m[i][j] == 1:
                edge.append([i, j])
    print(len(edge))

    # with open(os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w") as f0:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)


def load_data_deepDTnet(dataset_train="DTnet_train_0.8_0", dataset_test="DTnet_test_0.8_0"):
    dataset_dir = os.path.sep.join(['deepDTnet'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("deepDTnet_all")]), dtype=np.int32)
    # edge_train_pro = []
    # for i in edge_all:
    #     edge_train_pro.append([i[0], i[1] + 732])
    # with open(os.path.sep.join([dataset_dir, "edge_train_pro.txt"]), "w") as f0:
    #     for i in range(len(edge_train_pro)):
    #         s = str(edge_train_pro[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)
    # edge_test_pro = []
    # for i in edge_test:
    #     edge_test_pro.append([i[0], i[1] + 732])
    # with open(os.path.sep.join([dataset_dir, "edge_test_pro.txt"]), "w") as f0:
    #     for i in range(len(edge_test_pro)):
    #         s = str(edge_test_pro[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f0.write(s)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drugProtein.txt']), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    # val = np.zeros(len(edge_val))
    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1
            # val[i] = 1

    np.set_printoptions(threshold=np.inf)

    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()
    print("deepDTnet", H.size())  # 1915, 732
    drug_feat = torch.eye(732)
    prot_feat = torch.eye(1915)
    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'drugDisease.txt']), dtype=np.int32))  # 732, 440
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'proteinDisease.txt']), dtype=np.int32))  # 1915, 440

    return drugDisease, proteinDisease, drug_feat, prot_feat, H, H_T, edge_test, test


def generate_data_2(dataset_str="drug_target_interaction"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['deepDTnet'])
    # edge = np.genfromtxt("edges.txt", dtype=np.int32)
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    # print(edge)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)

    # drugs = []
    # targets = []
    # for i in edge:
    #     if i[0] not in drugs:
    #         drugs.append(i[0])
    #     if i[1] not in targets:
    #         targets.append(i[1])

    test_ration = [0.2]
    for d in test_ration:
        for a in (range(1)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) < len(edge_test) * 1:
                x1 = random.sample(range(0, 732), 1)[0]
                y1 = random.sample(range(0, 1915), 1)[0]
                if [x1, y1] not in edge.tolist() and [x1, y1] not in test_zeros:
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "DTnet_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "DTnet_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    # with open(os.path.sep.join([dataset_dir,  "DTnet_all.txt"]), "w") as f3:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f3.write(s)

