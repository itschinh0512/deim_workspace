import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
print(transforms)
