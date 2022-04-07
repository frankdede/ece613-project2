import torch
def sampler(input,size):
    return torch.nn.functional.interpolate(input, size=size, mode='nearest')
