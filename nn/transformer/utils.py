import copy
import torch.nn as nn


def replicate(m, N=6) -> nn.ModuleList:
    """Method to replicate the existing block to N set of blocks :param
    m: class inherited from nn.Module, mainly it is the encoder or
    decoder part of the architecture :param N: the number of stack, in
    the original paper they used 6 :return: a set of N blocks."""
    m_stack = nn.ModuleList([copy.deepcopy(m) for _ in range(N)])
    return m_stack
