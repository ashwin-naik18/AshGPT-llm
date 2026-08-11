import tiktoken
import torch

enc = tiktoken.get_encoding("o200k_base")

EOT_ID = enc.eot_token

def encode(text : str):
    return enc.encode(text)

def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
        
    return enc.decode(ids)

