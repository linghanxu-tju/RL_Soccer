import numpy as np
import logging
import torch
import torch.multiprocessing as mp
import torch.nn as nn



DarkBlue = '#051c2c'
LightBlue = '#3ba9f5'
LightGrey = '#989898'
Blue = '#2140e6'
BrightBlue = '#aae6f0'
Orange = '#FF7828'
Green = '#009926'
colors = [Blue, Green, Orange, LightGrey, LightBlue]


def load_my_state_dict(model, saved_state):
    pretrained_dict = torch.load(saved_state)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys and check the missed keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    missed_dict = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
    if len(missed_dict.keys()) >= 3:
        logging.warning("The Missed diction keys is larger than {}, Please check if the pretrain state and the code".format(len(missed_dict.keys())))
    new_dict = {**pretrained_dict, **missed_dict}
    print("Pretrained_dict Keys: {}".format(pretrained_dict.keys()))
    print("Missed dict Keys: {}, Will use the initial one".format(missed_dict.keys()))
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_dict)
    # 3. load the new state dict
    model.load_state_dict(new_dict)
    return model


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Counter():
    def __init__(self):
        self.val = mp.Value('i', 0)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def set(self, value):
        with self.lock:
            self.val.value = value

    def value(self):
        with self.lock:
            return self.val.value


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])