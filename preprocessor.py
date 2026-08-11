from google.colab import drive
drive.mount("/content/drive")



import os

save_dir = "/content/drive/MyDrive/SimpleStoriesChunks"
os.makedirs(save_dir, exist_ok=True)



import torch
from datasets import load_dataset
from tokenizer import encode
from datetime import datetime
from tokenizer import *

MAX_TOKEN = 2_000_000


print("Loading dataset...")
dataset = load_dataset("SimpleStories/SimpleStories")

print("Dataset Loaded..")

stories = dataset['train']['story']


split = int(len(stories) * 0.95)


train_stories = stories[:split]
val_stories = stories[split:]



print(len(stories))



encoded_chunk = []
chunk_id = 0


print("Saving Started..")

start = datetime.now()

print("Time of Start : ", start)

for i, story in enumerate(train_stories):

    ids = encode(story)
    
    ids.append(EOT_ID)
    
    encoded_chunk.extend(ids)

    if len(encoded_chunk) >= MAX_TOKEN:
        torch.save(
            torch.tensor(encoded_chunk, dtype=torch.int32),
            f"{save_dir}/chunk_{chunk_id}.pt"
        )
        encoded_chunk.clear()
        chunk_id += 1
        
        if chunk_id % 10 == 0:
            print(f"{chunk_id}th chunk saved..")

        
if len(encoded_chunk) > 0:
    torch.save(
        torch.tensor(encoded_chunk, dtype=torch.int32),
        f"{save_dir}/chunk_{chunk_id}.pt"
    )
    encoded_chunk.clear()
    

encoded_chunk = []
for i, story in enumerate(val_stories):
    
    ids = encode(story)

    ids.append(EOT_ID)
    
    encoded_chunk.extend(ids)

torch.save(
    torch.tensor(encoded_chunk, dtype=torch.int32),
    f"{save_dir}/val_chunk.pt"
)
    

print("Total Time Taken For Saving : ", datetime.now() - start)    

config = {
    "encoding" : "o200k_base",
    "num_chunks": chunk_id + 1,
    "train_size": len(train_stories),
    "val_size": len(val_stories),
    "dataset" : "SimpleStories"
}

torch.save(config, f"{save_dir}/config.pt")

print("Everything Saved..")