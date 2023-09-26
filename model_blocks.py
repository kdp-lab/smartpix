import torch
import torch.nn as nn
import numpy as np

def cascade_dims(input_dim, output_dim, depth):
    dimensions = [int(output_dim + (input_dim - output_dim)*(depth-i)/depth) for i in range(depth+1)]
    return dimensions

#%%%%%%% Classes %%%%%%%%#

class DNN_block(nn.Module):

    '''
    Similar to the embedding block, just takes care of progressive up/down shifting of layer size
    '''

    def __init__(self, input_dim, output_dim, dimensions, normalize_input):

        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None

        layers = []
        for iD, (dim_in, dim_out) in enumerate(zip(dimensions, dimensions[1:])):
            if dim_out != output_dim or iD != len(dimensions[1:])-1:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                    nn.LayerNorm(dim_out),
                    nn.ReLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead # .transpose(1,2)
            x = x #.transpose(1,2) # revert transpose
        return self.net(x)

class Model(nn.Module):

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim):

        super().__init__()

        # embed, In -> Out : J,C -> J,E
        self.embed = DNN_block(embed_input_dim, embed_dim, cascade_dims(embed_input_dim, embed_dim, embed_nlayers), normalize_input=False)

    def forward(self, x):

        x = self.embed(x)
        return x
