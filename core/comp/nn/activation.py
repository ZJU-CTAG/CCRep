from torch import nn

from utils import GlobalLogger as mylogger


def build_activation(activation_name, **kwargs):
    if activation_name not in activation_switch:
        mylogger.warning('build_activation]',
                         f'Unrecognized activation: {activation_name}. Return identity instead')
    return activation_switch.get(activation_name, _identity)(**kwargs)


def _tanh(**kwargs):
    return nn.Tanh()


def _sigmoid(**kwargs):
    return nn.Sigmoid()


def _relu(**kwargs):
    return nn.ReLU()


def _identity(**kwargs):
    return nn.Identity()


activation_switch = {
    'tanh': _tanh,
    'sigmoid': _sigmoid,
    'relu': _relu,
    None: _identity,
    'identity': _identity,
    'none': _identity,
}