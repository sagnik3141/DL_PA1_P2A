import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    This class defines the model used for training.

    input_dim: Dimension of the input data
    num_classes: Number of classes to be classified
    num_nodes_h1: Number of nodes in hidden layer 1
    num_nodes_h2: Number of nodes in hidden layer 2
    """

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim
        self.num_classes = args.num_classes
        self.num_nodes_h1 = args.num_nodes_h1
        self.num_nodes_h2 = args.num_nodel_h2

        
        self.linear1 = nn.Linear(self.input_dim, self.num_nodes_h1)
        self.tanh1 = nn.Tanh(inplace = True)
        self.linear2 = nn.Linear(self.num_nodes_h1, self.num_nodes_h2)
        self.tanh2 = nn.Tanh(inplace = True)
        self.out_layer = nn.Linear(self.num_nodes_h2, self.num_classes)
        
    def forward(self, x):
        assert x.shape[-1] == self.input_dim, "Please provide data of appropriate dimensions."

        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.out_layer(x)

        return x