import torch

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.memory_allocated(0))
    print(torch.cuda.memory_reserved(0))