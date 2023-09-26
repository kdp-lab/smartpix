import torch
import torch.nn as nn
import numpy as np

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
            print(x.shape)
            x = self.input_bn(x.transpose(1,2)) # must transpose because batchnorm expects N,C,L rather than N,L,C like multihead
            x = x.transpose(1,2) # revert transpose
        return self.net(x)

class Model(nn.Module):

    def __init__(self, embed_input_dim, embed_nlayers, embed_dim):

        super().__init__()

        # embed, In -> Out : J,C -> J,E
        self.embed = DNN_block(embed_input_dim, embed_dim, [embed_input_dim] + (embed_nlayers+1)*[embed_dim], normalize_input=True)

    def forward(self, x):

        x = self.embed(x)
        return x
