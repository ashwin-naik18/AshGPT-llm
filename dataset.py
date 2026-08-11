from typing import Any

import torch
import random
from torch.utils.data import Dataset

class ChunkDataset(Dataset):
    def __init__(self, chunk, block_size):
        self.chunk = chunk
        self.block_size = block_size
        
    
    def __len__(self):
        return len(self.chunk) - self.block_size - 1 
    
    def __getitem__(self, index) -> Any:
        x = self.chunk[index : index + self.block_size]
        y = self.chunk[index + 1 : index + 1 + self.block_size]
        
        return x, y