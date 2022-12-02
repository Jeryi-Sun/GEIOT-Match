import torch
from torch_geometric.data import Data
import json
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.data import HeteroData
from utils.snippets import *
import torch_geometric.transforms as T
"path global variables"
ELAM_save_edges = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/Edge_dataset.json"
ELAM_save_edges_dict = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/Edge_Dict_dataset.json"



id2vocab_law_path = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/id2vocab_law.pickle"
id2vocab_doc_path = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/id2vocab_doc.pickle"
id2vocab_key_path = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/id2vocab_key.pickle"
ELAM_dataset_path = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM_dataset.json"


"""
from torch_geometric.data import HeteroData

data = HeteroData()

data['law_article'].x = ... # [num_papers, num_features_paper]
data['law_cases'].x = ... # [num_authors, num_features_author]
data['key_paragraph'].x = ... # [num_institutions, num_features_institution]

data['law_cases', 'cites', 'law_article'].edge_index = ... # [2, num_edges_cites]
data['law_cases', 'contains', 'key_paragraph'].edge_index = ... # [2, num_edges_writes]

data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
data['author', 'writes', 'paper'].edge_attr = ... # [num_edges_writes, num_features_writes]
data['author', 'affiliated_with', 'institution'].edge_attr = ... # [num_edges_affiliated, num_features_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_attr = ... # [num_edges_topic, num_features_topic]
"""

def make_hetero_Data(args):
    """
    edge_index 需要分一下
    transforms 一下
    import torch_geometric.transforms as T

    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    data = T.NormalizeFeatures()(data)
    :param args:
    :return:
    """
    with open(id2vocab_law_path, 'rb') as f:
        id2vocab_law = pickle.load(f)
    with open(id2vocab_doc_path, 'rb') as f:
        id2vocab_doc = pickle.load(f)
    with open(id2vocab_key_path, 'rb') as f:
        id2vocab_key = pickle.load(f)

    with open(ELAM_save_edges_dict, 'r') as f:
        edges_dict = json.load(f)


    law_article, law_cases, key_paragraph = [], [], []
    for k in id2vocab_doc:
        law_cases.append(id2vocab_doc[k][-1][0])
    for k in id2vocab_law:
        law_article.append(id2vocab_law[k][-1][0])
    for k in id2vocab_key:
        key_paragraph.append(id2vocab_key[k][-1][0])
    law_article, law_cases, key_paragraph = torch.tensor(law_article, dtype=torch.float), torch.tensor(law_cases, dtype=torch.float), \
                                            torch.tensor(key_paragraph, dtype=torch.float)
    case2law = torch.tensor(edges_dict["case2law"])
    case2key = torch.tensor(edges_dict["case2key"])
    law2law = torch.tensor(edges_dict["law2law"])
    graph_data = HeteroData()
    graph_data["law_article"].x = law_article  # [num_article, num_features_paper]
    graph_data['law_cases'].x = law_cases  # [num_cases, num_features_author]
    graph_data['key_paragraph'].x = key_paragraph  # [num_key_paragraph, num_features_institution]

    graph_data['law_cases', 'cites', 'law_article'].edge_index = case2law.t().contiguous()  # [2, num_edges_cites]
    graph_data['law_cases', 'contains', 'key_paragraph'].edge_index = case2key.t().contiguous()  # [2, num_edges_contrains]
    graph_data['law_article', 'level', 'law_article'].edge_index = law2law.t().contiguous()  # [2, num_edges_level]
    graph_data = T.ToUndirected()(graph_data)

    print("++++++++++++graph information++++++++")
    print("graph data: ", graph_data)
    print("has_self_loops: ", graph_data.has_self_loops())
    print("is_directed: ", graph_data.is_directed())
    return graph_data



def make_graph_data(args):
    with open(ELAM_save_edges, 'r') as f:
        edge_index = json.load(f)
    edge_index = torch.tensor(edge_index)
    with open(id2vocab_law_path, 'rb') as f:
        id2vocab_law = pickle.load(f)
    with open(id2vocab_doc_path, 'rb') as f:
        id2vocab_doc = pickle.load(f)
    with open(id2vocab_key_path, 'rb') as f:
        id2vocab_key = pickle.load(f)

    node_x = []
    for k in id2vocab_doc:
        node_x.append(id2vocab_doc[k][-1][0])
    for k in id2vocab_law:
        node_x.append(id2vocab_law[k][-1][0])
    for k in id2vocab_key:
        node_x.append(id2vocab_key[k][-1][0])
    node_x = torch.tensor(node_x, dtype=torch.float)
    graph_data = Data(x=node_x, edge_index=edge_index.t().contiguous())
    print("++++++++++++graph information++++++++")
    print("graph data: ", graph_data)
    print("num_node_feature: ", graph_data.num_node_features)
    print("has_isolated_nodes: ", graph_data.has_isolated_nodes())
    print("has_self_loops: ", graph_data.has_self_loops())
    print("is_directed: ", graph_data.is_directed())
    return graph_data


if __name__ == '__main__':
    make_hetero_Data(None)