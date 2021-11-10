import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import random
import scipy.io
from sklearn.decomposition import non_negative_factorization
import pandas as pd

random.seed(3)

def pre_processed_kegg_kg():
    dataset_dir = os.path.sep.join(['kegg'])
    dti = pd.read_table(os.path.sep.join([dataset_dir, 'dt_kegg_med.txt']), header=None)
    # print(dti)
    drug_list = set(dti[0])
    # print(len(drug_list))  # 4284
    target_list = set(dti[2])
    # print(len(target_list))  # 945
    kg = pd.read_table(os.path.sep.join([dataset_dir, 'kegg_kg.txt']), header=None)
    kg_drug = kg[kg[1].str.contains('PATHWAY_DRUG')]
    # print(kg_drug)
    drug_pathway = []
    for row in kg_drug.itertuples():
        if getattr(row, '_3') in drug_list:
            drug_pathway.append([getattr(row, '_1'), getattr(row, '_3')])
    # print("drug_pathway", drug_pathway)

    kg_gene = kg[kg[1].str.contains('PATHWAY_GENE')]
    gene_pathway = []
    for row in kg_gene.itertuples():
        if getattr(row, '_3') in target_list:
            gene_pathway.append([getattr(row, '_1'), getattr(row, '_3')])
    # print("gene_pathway", gene_pathway)

    drug_dict = {}
    for index, i in enumerate(drug_list):
        drug_dict[i] = index
    print("drug_dict", drug_dict)

    target_dict = {}
    for index, i in enumerate(target_list):
        target_dict[i] = index
    # print("target_dict", target_dict)

    pathway1 = np.array(drug_pathway)[:, 0]
    pathway2 = np.array(gene_pathway)[:, 0]
    pathway = set(np.concatenate((pathway1, pathway2), axis=0))
    # print(len(pathway))  # 105

    pathway_dict = {}
    for index, i in enumerate(pathway):
        pathway_dict[i] = index
    # print("pathway_dict", pathway_dict)

    drug_pathway_processed = []
    for i in drug_pathway:
        if i[0] in pathway_dict:
            drug_pathway_processed.append([pathway_dict[i[0]], drug_dict[i[1]]])
    # print("drug_pathway_processed", drug_pathway_processed)
    # print(len(drug_pathway_processed))  # 7087

    target_pathway_processed = []
    for i in gene_pathway:
        if i[0] in pathway_dict:
            target_pathway_processed.append([pathway_dict[i[0]], target_dict[i[1]]])
    # print("target_pathway_processed", target_pathway_processed)
    # print(len(target_pathway_processed))  # 3390

    drug_target_processed = []
    for row in dti.itertuples():
        drug_target_processed.append([drug_dict[getattr(row, '_1')], target_dict[getattr(row, '_3')]])
    # print("drug_target_processed", drug_target_processed)
    # print(len(drug_target_processed))  # 12112

    H_drug_pathway = np.zeros((len(drug_dict), len(pathway_dict)), dtype=np.int32)
    for i in drug_pathway_processed:
        H_drug_pathway[i[1], i[0]] = 1


    H_target_pathway = np.zeros((len(target_dict), len(pathway_dict)), dtype=np.int32)
    for i in target_pathway_processed:
        H_target_pathway[i[1], i[0]] = 1
    # print(len(H_target_pathway), len(H_target_pathway[0]))  # 945*105

    with open(os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w") as f0:
        for i in range(len(drug_target_processed)):
            s = str(drug_target_processed[i]).replace('[', ' ').replace(']', ' ')
            s = s.replace("'", ' ').replace(',', '') + '\n'
            f0.write(s)

    np.savetxt(os.path.sep.join([dataset_dir, "H_drug_pathway.txt"]), H_drug_pathway)
    np.savetxt(os.path.sep.join([dataset_dir, "H_target_pathway.txt"]), H_target_pathway)

    disease_drug = kg[kg[1].str.contains('DRUG_EFFICACY_DISEASE')]
    # print(disease_drug)
    drug_disease = []
    for row in disease_drug.itertuples():
        if getattr(row, '_1') in drug_list:
            drug_disease.append([getattr(row, '_1'), getattr(row, '_3')])
    print("drug_disease", drug_disease)

    disease_target = kg[kg[1].str.contains('GENE_DISEASE')]
    # print(disease_target)
    target_disease = []
    for row in disease_target.itertuples():
        if getattr(row, '_1') in target_list:
            target_disease.append([getattr(row, '_1'), getattr(row, '_3')])
    print("target_disease", target_disease)

    disease1 = np.array(drug_disease)[:, -1]
    disease2 = np.array(target_disease)[:, -1]
    disease = set(np.concatenate((disease1, disease2), axis=0))
    # print(len(disease))  # 360

    disease_dict = {}
    for index, i in enumerate(disease):
        disease_dict[i] = index
    print("disease_dict", disease_dict)

    drug_disease_processed = []
    for i in drug_disease:
        if i[1] in disease_dict:
            drug_disease_processed.append([disease_dict[i[1]], drug_dict[i[0]]])
    # print("drug_disease_processed", drug_disease_processed)
    # print(len(drug_disease_processed))  # 365

    target_disease_processed = []
    for i in target_disease:
        if i[1] in disease_dict:
            target_disease_processed.append([disease_dict[i[1]], target_dict[i[0]]])
    # print("target_disease_processed", target_disease_processed)
    # print(len(target_disease_processed))  # 433

    H_drug_disease = np.zeros((len(drug_dict), len(disease_dict)), dtype=np.int32)
    for i in drug_disease_processed:
        H_drug_disease[i[1], i[0]] = 1
    # print(H_drug_disease)
    # print(len(H_drug_disease), len(H_drug_disease[0]))  # 4284*360

    H_target_disease = np.zeros((len(target_dict), len(disease_dict)), dtype=np.int32)
    for i in target_disease_processed:
        H_target_disease[i[1], i[0]] = 1
    # print(len(H_target_disease), len(H_target_disease[0]))  # 945*360

    np.savetxt(os.path.sep.join([dataset_dir, "H_drug_disease.txt"]), H_drug_disease)
    np.savetxt(os.path.sep.join([dataset_dir, "H_target_disease.txt"]), H_target_disease)


# pre_processed_kegg_kg()


def load_data_KEGG_MED(dataset_train="kegg_train_0.8_0", dataset_test="kegg_test_0.8_0"):
    dataset_dir = os.path.sep.join(['KEGG_MED'])

    # build incidence matrix
    edge_train = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_train)]), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format("kegg_all")]), dtype=np.int32)
    edge_test = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_test)]), dtype=np.int32)
    # print('edge_test', len(edge_test) / 2)
    # edge_val = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_val)]), dtype=np.int32)
    # print(edge_train)

    i_m = np.genfromtxt(os.path.sep.join([dataset_dir, 'drug_target_interaction.txt']), dtype=np.int32)

    H_T = np.zeros((4284, 945), dtype=np.int32)
    H_T_all = np.zeros((4284, 945), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1
    # np.savetxt(os.path.sep.join([dataset_dir,  "drugProtein.txt"]), H_T_all)
    # with open(os.path.sep.join([dataset_dir,  "drugProtein.txt"]), "w") as f3:
    #     for i in range(len(H_T_all)):
    #         s = str(H_T_all[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f3.write(s)
    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1


    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()


    print("KEGG_MED", H.size())  # 945, 4284
    drug_feat1 = torch.eye(4284)
    prot_feat1 = torch.eye(945)
    drugpathway = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_drug_pathway.txt']), dtype=np.int32))
    proteinpathway = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_target_pathway.txt']), dtype=np.int32))
    drugDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_drug_disease.txt']), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.sep.join([dataset_dir, 'H_target_disease.txt']), dtype=np.int32))
    # print(drugDisease)
    # print(drugDisease.size())  # 4284 360
    # print(proteinDisease.size())  # 945 360


    pos2 = []
    for i in range(len(H_T)):
        if 1 in H_T[i]:
            pos2.append(1)
        else:
            pos2.append(0)
    print(pos2)

    print(edge_test)
    pos_island = []
    pos_ = []
    neg_island = []
    neg_ = []
    for i in range(len(edge_test)):
        if i < len(edge_test) // 2:
            if pos2[edge_test[i][0]] == 1:
                pos_.append(i)
            else:
                pos_island.append(i)
        else:
            if pos2[edge_test[i][0]] == 1:
                neg_.append(i)
            else:
                neg_island.append(i)


    return drugDisease, proteinDisease, drug_feat1, prot_feat1, H, H_T, edge_test, test


def generate_data_2(dataset_str="drug_target_interaction"):
    # 将数据集分为训练集，测试集
    dataset_dir = os.path.sep.join(['KEGG_MED'])
    edge = np.genfromtxt(os.path.sep.join([dataset_dir, '{}.txt'.format(dataset_str)]), dtype=np.int32)  # dtype='U75'
    # print(edge)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    # print(data)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())
    # print(edge_shuffled)

    drugs = []
    targets = []
    for i in edge:
        if i[0] not in drugs:
            drugs.append(i[0])
        if i[1] not in targets:
            targets.append(i[1])

    test_ration = [0.2]
    for d in test_ration:
        for a in (range(1)):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d): (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[(a + 1) * int(len(edge_shuffled) * d):]

            test_zeros = []
            while len(test_zeros) < len(edge_test):
                x1 = random.sample(range(0, len(drugs)), 1)[0]
                y1 = random.sample(range(0, len(targets)), 1)[0]
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros:
                    test_zeros.append([x1, y1])
            # print(test_zeros)
            edge_test = edge_test + test_zeros

            with open(os.path.sep.join([dataset_dir, "kegg_train_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f0:
                for i in range(len(edge_train)):
                    s = str(edge_train[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f0.write(s)

            with open(os.path.sep.join([dataset_dir, "kegg_test_{ratio}_{fold}.txt".format(ratio=d, fold=a)]), "w") as f1:
                for i in range(len(edge_test)):
                    s = str(edge_test[i]).replace('[', ' ').replace(']', ' ')
                    s = s.replace("'", ' ').replace(',', '') + '\n'
                    f1.write(s)

    # with open(os.path.sep.join([dataset_dir,  "kegg_all.txt"]), "w") as f3:
    #     for i in range(len(edge)):
    #         s = str(edge[i]).replace('[', ' ').replace(']', ' ')
    #         s = s.replace("'", ' ').replace(',', '') + '\n'
    #         f3.write(s)

# generate_data_2()


