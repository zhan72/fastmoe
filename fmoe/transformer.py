r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import FMoE
from .linear import FMoELinear
from .fastermoe.config import switch_from_env


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class _Expert_linear(nn.Module):
    def __init__(self, num_expert, d_model, d_outsize, rank=0):
        super().__init__()
        self.linear = FMoELinear(num_expert, d_model, d_outsize, bias=True, rank=rank)

    def forward(self, inp, fwd_expert_count):
        x = self.linear(inp, fwd_expert_count)
        return x

class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        d_outsize=None,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        use_linear=False, 
        **kwargs
    ):
        if d_outsize==None:
            d_outsize = d_model
        self.d_outsize = d_outsize
        def one_expert(d_model):
            if use_linear==False:
                return _Expert(1, d_model, d_hidden, d_outsize , activation, rank=0)
            else:
                return _Expert_linear(1, d_model, d_outsize, rank=0)
        
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = list(inp.shape)
        assert original_shape[-1] == self.d_model
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        original_shape[-1] = self.d_outsize 
        return output.reshape(original_shape)
