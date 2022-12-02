from torch_geometric.nn import HGTConv, Linear
import torch
import torch.nn as nn
class Match_HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, args):
        super().__init__()
        self.dataset = data
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = torch.nn.Sequential(Linear(-1, hidden_channels),
                                                           torch.nn.ReLU(),
                                                           Linear(hidden_channels, hidden_channels))

        self.convs = torch.nn.ModuleList()
        self.convs_lin = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='mean')
            self.convs.append(conv)
            lin_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                lin_dict[node_type] = torch.nn.Sequential(Linear(-1, hidden_channels),
                                                               torch.nn.ReLU(),
                                                               Linear(hidden_channels, hidden_channels))
                self.convs_lin.append(lin_dict)

        self.lin_Seq = nn.Sequential(Linear(4*hidden_channels, hidden_channels),
                                     nn.ReLU(),
                                     nn.Dropout(p=args.dropout_rate),
                                     Linear(hidden_channels, out_channels))

    def forward(self, doc_id_A, doc_id_B):
        x_dict, edge_index_dict = self.dataset.x_dict, self.dataset.edge_index_dict
        x_dict_features = {}
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        for k, conv in enumerate(self.convs):
            if k == 0:
                x_dict_features = conv(x_dict, edge_index_dict)
            else:
                x_dict_features = conv(x_dict_features, edge_index_dict)

            for node_type, x in x_dict_features.items():
                x_dict_features[node_type] = self.convs_lin[k][node_type](x).relu_()  # add residual connections

        return x_dict_features['law_cases'][doc_id_A], x_dict_features['law_cases'][doc_id_B]
