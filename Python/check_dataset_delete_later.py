from ogb.linkproppred import PygLinkPropPredDataset
import torch
import numpy as np
from ogb.linkproppred import Evaluator
d_name = "ogbl-ddi"

# load dataset (first time running will download zip file otherwise existing file in dataset folder will be used)
dataset = PygLinkPropPredDataset(name = d_name)
# split loaded data
split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
print(len(train_edge['edge']))
print('done')