import torch
from torch.nn.modules.module import Module

"""
    KL divergences of node embedding and hyperedge embedding 
"""
class kl_loss(Module):
    def __init__(self, num_nodes, num_edges):
        super(kl_loss, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, z_node_log_std, z_node_mean, z_edge_log_std, z_edge_mean):
        kl_node = - (0.5 / self.num_nodes) * torch.mean(torch.sum(
            1 + 2 * z_node_log_std - torch.pow(z_node_mean, 2) - torch.pow(torch.exp(z_node_log_std), 2),
            1))

        kl_edge = - (0.5 / self.num_edges) * torch.mean(
            torch.sum(
                1 + 2 * z_edge_log_std - torch.pow(z_edge_mean, 2) - torch.pow(torch.exp(z_edge_log_std), 2), 1))

        kl = kl_node + kl_edge

        return kl
