import torch
import torch.nn.functional as F 
from config import *
from model import *
from tokenizer import *

save_dir = "/content/drive/MyDrive/SimpleStoriesChunks"

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = enc.n_vocab

model = GPT(
    vocab_size, EMBEDDING_DIM, BLOCK_SIZE, NUM_HEAD, NUM_LAYER
).to(device=device)

check_point = torch.load(
    f"{save_dir}/best_model.pt",
    map_location=device
)

model.load_state_dict(
    check_point['model']
)

model.eval()


def generate(prompt, max_token = 100, temparature = 1.0, top_k = 50):
    
    with torch.no_grad():
        ids = encode(prompt)
        
        x = torch.tensor(
            ids, 
            dtype= torch.long, 
            device=device
        ).unsqueeze(0)
        
        for _ in range(max_token):
            x = x[:, -model.block_size:]
            
            
            logits, _ = model(x)
            
            
            logits = logits[:, -1, :]
            
            logits = logits / temparature
            
            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    top_k
                )
                
            logits[logits < values[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim = -1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
            if next_token.item() == EOT_ID:
                break
            
        
        return decode(x.squeeze(0))