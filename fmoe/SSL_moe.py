from .transformer import FMoETransformerMLP
import sys
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def SSLfmoefy(moe_num_experts=None,hidden_size=328,hidden_hidden_size=None,d_outsize=None,top_k=1,use_linear=False):
    activation = nn.PReLU()
    model = FMoETransformerMLP(num_expert=moe_num_experts, d_model=hidden_size, d_hidden=hidden_hidden_size, d_outsize=d_outsize ,top_k=top_k, activation=activation,use_linear=use_linear)
    return model

        
   