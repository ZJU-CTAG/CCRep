from torch import nn

from core.comp.nn.activation import build_activation

def mlp_block(in_dim, out_dim, activation, dropout):
    blocks = [nn.Linear(in_dim, out_dim)]

    activation_block = build_activation(activation)
    blocks.append(activation_block)

    if dropout is not None:
        blocks.append(nn.Dropout(dropout))

    return nn.Sequential(*blocks)