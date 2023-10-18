import torch


def l2norm(inputs, dim=-1):
  # inputs: (batch, dim_ft)
  norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
  inputs = inputs / norm.clamp(min=1e-10)
  return inputs






