import pickle
import os
from collections import OrderedDict


def state_dict_trans(state_dict, file_path=None):
    new_dict = OrderedDict()
    for param_tensor in state_dict:
        new_dict[param_tensor] = state_dict[param_tensor].cpu().numpy()
    if file_path:
        f = open(file_path, "wb")
        pickle.dump(new_dict, f)
        f.close()
    return new_dict


def load_trajectory(save_dir):
    trajectorys = []
    for filename in os.listdir(save_dir):
        file = os.path.join(save_dir, filename)
        f = open(file, "rb")
        trajectory = pickle.load(f)
        trajectorys.extend(trajectory)
    return trajectorys
