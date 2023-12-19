import torch
print(torch.__version__);
print(torch.cuda.is_available())
from torch.backends import cudnn
print(cudnn.is_available())