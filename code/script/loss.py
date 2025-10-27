import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from script.config import args
from script.hgcn.manifolds import PoincareBall
from torch_geometric.utils import negative_sampling
from script.utils.util import logger

device = args.device

EPS = 1e-15
MAX_LOGVAR = 10


class ReconLoss(nn.Module):
    def __init__(self, args):
        super(ReconLoss, self).__init__()
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = PoincareBall()
        self.use_hyperdecoder = True

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdeoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index) + EPS).mean()
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder

        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

