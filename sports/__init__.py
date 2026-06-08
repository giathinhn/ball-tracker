import torch

# PyTorch float8 compatibility fix for transformers
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
