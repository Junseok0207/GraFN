import torch
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP
import torch.nn as nn
import torch.nn.functional as F

# Borrowed from BGRL
class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm_mm=0.99):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            #layers.append( (nn.Dropout(drop_rate), 'x -> x'), )
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)
            layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()
