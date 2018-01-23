import torch
from utils import V_Re_ID_Dataset
import models
import sys


path = sys.argv[1]

Dataset = V_Re_ID_Dataset(path)
print(len(Dataset))

