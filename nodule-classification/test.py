from resnet_attn import *
from kflod import kfold
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.densenet import densenet121
from torch.optim.adam import Adam
import numpy as np
import random
import torch

def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

model = LocalGlobalNetwork()
print(torch.load(f"./results/LocalGlobalNetwork_result")['all_pred'])
# model.load_state_dict(torch.load(f"./results/LocalGlobalNetwork_result"))
