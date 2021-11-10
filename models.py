import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from layers import *
import numpy as np


class HETE1(nn.Module):
    def __init__(self, num_in_node, num_in_edge, num_hidden1, num_out, num_out1=64):  # 1915, 732, 512, 128
        super(HETE1, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)

        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self.hgnn_node1 = HGNN2(num_in_node, num_hidden1, num_out, num_in_node, num_in_node)

        self.hgnn_hyperedge1 = HGNN2(num_in_edge, num_hidden1, num_out, num_in_edge, num_in_edge)

        self.act = torch.sigmoid

        self.attention1 = self_Attention(num_in_node, num_out, num_out1)
        self.attention2 = self_Attention(num_in_node, num_out, num_out1)
        self.attention3 = self_Attention(num_in_edge, num_out, num_out1)
        self.attention4 = self_Attention(num_in_edge, num_out, num_out1)


    def sample_latent(self, z_node, z_hyperedge):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).float()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)  # sigma
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).float()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_  # Reparameterization trick
        else:
            return self.z_node_mean, self.z_edge_mean

    # def forward(self, G1, G2, G3, G4, drug_vec, protein_vec, H, H_T):
    def forward(self, G1, G2, drug_vec, protein_vec, H, H_T):
        # side embedding
        drug_feature = self.hgnn_hyperedge1(drug_vec, G1)
        # print(drug_feature.size())
        protein_feature = self.hgnn_node1(protein_vec, G2)
        # print(protein_feature.size())
        # pathway_protein = self.hgnn_node2(protein_vec, G4)
        # print(effect_feature1.size())
        # pathway_drug = self.hgnn_hyperedge2(drug_vec, G3)
        # print(effect_feature1.size())
        # effect_feature2 = self.hgnn_node2(protein_vec, G4)
        # alpha

        # key embedding
        # z_node_encoder = self.node_encoders1(torch.cat((H, G2), 1))
        z_node_encoder = self.node_encoders1(H)

        # z_hyperedge_encoder = self.hyperedge_encoders1(torch.cat((H_T, G1), 1))
        z_hyperedge_encoder = self.hyperedge_encoders1(H_T)

        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)

        # fuse layer
        # a_protein_feature = torch.exp(self.attention1(protein_feature))
        # a_z_node_s = torch.exp(self.attention2(self.z_node_s))
        # z_node = (a_protein_feature / (a_protein_feature + a_z_node_s)) * protein_feature + (a_z_node_s / (a_protein_feature + a_z_node_s)) * self.z_node_s
        # z_node = protein_feature  + self.attention2(self.z_node_s)
        # z_node = self.attention1(protein_feature) + self.z_node_s
        # z_node = protein_feature1 + self.z_node_s
        # z_node = torch.cat((self.z_node_s, protein_feature), 1)
        z_node = self.z_node_s

        # a_drug_feature = torch.exp(self.attention3(drug_feature))
        # a_z_hyperedge_s = torch.exp(self.attention4(self.z_hyperedge_s))
        # z_hyperedge = (a_drug_feature / (a_drug_feature + a_z_hyperedge_s)) * drug_feature + (a_z_hyperedge_s / (a_drug_feature + a_z_hyperedge_s)) * self.z_hyperedge_s

        # z_hyperedge = self.attention3(drug_feature) + self.attention4(self.z_hyperedge_s)
        # z_hyperedge = drug_feature  # + self.z_hyperedge_s
        # z_hyperedge = torch.cat((self.z_hyperedge_s, drug_feature), 1)
        z_hyperedge = self.z_hyperedge_s

        # reconstruction = self.decoder2(z_node, z_hyperedge)

        # recover = torch.sigmoid(torch.cat((self.z_node_mean, protein_feature), 1).mm(torch.cat((self.z_edge_mean, drug_feature), 1).t()))
        # node_embedding = self.z_node_mean.cpu().detach().numpy()
        # edge_embedding = self.z_edge_mean.cpu().detach().numpy()
        # np.savetxt('node_embedding.txt', node_embedding)
        # np.savetxt('edge_embedding.txt', edge_embedding)
        reconstruction1 = self.decoder2(z_node, z_hyperedge)
        # reconstruction2 = self.decoder2(protein_feature, drug_feature)
        # reconstruction2 = reconstruction1
        # recover = torch.sigmoid(self.z_node_mean.mm(self.z_edge_mean.t()))
        recover = self.z_node_mean.mm(self.z_edge_mean.t())
        # return reconstruction1, reconstruction2, recover
        return reconstruction1, recover



