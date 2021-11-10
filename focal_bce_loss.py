import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
    focal bce loss because of the imbalance of the sample
"""
class Focal_Loss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2, logits=True, reduce=True):
        super(Focal_Loss, self).__init__()

        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if self.logits:
            # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight, reduce=True)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, pos_eight=self.pos_weight, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt) ** (self.gamma) * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

