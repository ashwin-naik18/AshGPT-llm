# AshGPT — A GPT-style Language Model Built from Scratch

A minimal, from-scratch implementation of a GPT-style transformer language model in PyTorch, trained on the [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) dataset.

This project implements the core building blocks of a decoder-only transformer — multi-head self-attention, feed-forward networks, layer normalization, dropout, and residual connections — without relying on high-level transformer libraries, as a way of understanding how GPT-style models work under the hood.

## The Problem in v3

v3 turned training into a proper, resumable, monitored pipeline, but a few things were still holding it back once training actually got going:

- **Word-level tokenizer** — the vocabulary was built by splitting on whitespace, so every unique word became its own token. This produced a huge, sparse vocabulary, sent any unseen or misspelled word straight to `<UNK>`, and wasted embedding capacity on words that share most of their structure (e.g. "run", "running", "runner" were three unrelated tokens).
- **No dropout anywhere in the model** — attention weights and feed-forward outputs passed through untouched, leaving the model more prone to overfitting on a small dataset.
- **No final layer norm before the output head** — logits were computed directly from the last transformer block's output, deviating from the standard pre-norm GPT design and making training less stable.
- **Default weight initialization** — linear and embedding layers relied on PyTorch's default init instead of the small-std normal initialization GPT-style models are typically trained with, which can slow down and destabilize early training.
- **Synchronous chunk loading** — each data chunk was loaded from disk only when training reached it, so the GPU sat idle waiting on I/O between chunks instead of training continuously.
- **Chunking by story count, not token count** — chunks were built from a fixed number of stories, so their actual token length varied depending on how long those stories happened to be, making chunk size (and therefore memory/compute per chunk) unpredictable.
- **Greedy-ish multinomial sampling only** — generation always sampled from the full softmax distribution with no way to control randomness or restrict to the most likely tokens, which could produce incoherent output.

## What Changed in v4 (AshGPT)

| | v3 | v4 (AshGPT) |
|---|---|---|
| Tokenizer | Custom word-level vocab built from the corpus, with `<UNK>` fallback | BPE tokenization via `tiktoken`'s `o200k_base` encoding — a fixed, pretrained subword vocabulary with no OOV problem |
| Chunking strategy | Fixed number of stories per chunk (10,000), so token count per chunk varied | Fixed token budget per chunk (2M tokens), giving consistent, predictable chunk sizes regardless of story length |
| Storage dtype | Tokens stored as `int64` | Tokens stored as `int32`, roughly halving on-disk and in-memory footprint |
| Batching | Manual random-window sampling (`get_batch`) reading directly from a raw tensor | Dedicated `ChunkDataset` (`torch.utils.data.Dataset`) fed through a `DataLoader` with shuffling and pinned memory, for both train and validation |
| Data loading | Synchronous — training blocked while each new chunk loaded from disk | Chunks prefetched in a background thread (`ThreadPoolExecutor`) while the current chunk trains, hiding I/O latency |
| Regularization | No dropout | Dropout (0.1) applied in attention weights, the multi-head output projection, and the feed-forward block |
| Activation function | ReLU in the feed-forward network | GELU, matching standard GPT-style architectures |
| Output normalization | Logits computed directly from the last block's output | Final `LayerNorm` (`ln_f`) applied before the LM head, matching the standard pre-norm GPT design |
| Weight initialization | PyTorch default init | Explicit GPT-style init — linear and embedding weights drawn from `N(0, 0.02)`, biases zeroed |
| Mixed precision | Always enabled, including on CPU | Conditionally enabled only when running on CUDA, avoiding unsupported autocast/scaler behavior on CPU |
| Generation control | Plain multinomial sampling over the full distribution | Temperature scaling and top-k filtering, for controllable, higher-quality sampling |
| Parameter counting | Manual `sum(p.numel() ...)` inline in the training script | `model.count_parameters()` method on the `GPT` class itself |

In short: v3 built a training pipeline that could run reliably end-to-end; v4 makes what's actually being trained better — a real subword tokenizer instead of a brittle word-level vocab, a properly regularized and normalized architecture matching standard GPT design, non-blocking data loading so the GPU stays fed, and controllable generation instead of raw sampling.

## Features

- **Custom transformer implementation** — self-attention heads, multi-head attention, feed-forward blocks, and transformer blocks written from scratch (no `nn.Transformer` or HuggingFace models)
- **BPE tokenization** via `tiktoken` (`o200k_base`), avoiding out-of-vocabulary issues entirely
- **Causal (masked) self-attention** with a precomputed, buffered mask
- **Dropout regularization** throughout attention and feed-forward layers
- **GPT-style architecture details** — GELU activations, pre-norm transformer blocks, a final layer norm before the output head, and small-std weight initialization
- **Token-budgeted chunked preprocessing** with a proper story-level train/validation split
- **Background chunk prefetching** to keep the GPU fed between chunks
- **`Dataset`/`DataLoader`-based batching** with shuffling and pinned memory
- **Mixed-precision training** (`torch.amp`) with gradient scaling and clipping, safely disabled on non-CUDA devices
- **Cosine annealing learning rate schedule**
- **`torch.compile`** for faster training throughput
- **Full checkpoint state saving** — model, optimizer, scaler, scheduler, epoch, and best loss
- **Live training telemetry** — loss, learning rate, progress percentage, and ETA
- **Controllable text generation** — temperature and top-k sampling in an easy-to-call `generate()` function

## Project Structure

```
.
├── config.py         # All hyperparameters in one place
├── tokenizer.py       # BPE tokenizer (tiktoken, o200k_base) — encode/decode + EOT token
├── dataset.py         # ChunkDataset: windows a token chunk into (x, y) training pairs
├── preprocessor.py   # Mounts Drive, downloads SimpleStories, tokenizes, splits train/val, saves token-budgeted chunks
├── model.py           # GPT architecture: attention heads, transformer blocks, LM head
├── train.py            # Training loop: async prefetching, mixed precision, LR schedule, checkpointing
├── generate.py       # Loads the trained model and generates text with temperature/top-k sampling
└── README.md
```

## Architecture

The model is a decoder-only transformer, closely following the standard GPT design:

| Component      | Description                                              |
|-----------------|-----------------------------------------------------------|
| Token Embedding | Maps token ids to dense vectors                          |
| Position Embedding | Learned positional encodings                          |
| Attention Head  | Scaled dot-product attention with a causal mask and dropout |
| Multi-Head Attention | Multiple attention heads concatenated, projected, and dropped out |
| Feed-Forward Network | 2-layer MLP with GELU, 4x expansion, and dropout |
| Transformer Block | Pre-LayerNorm attention + FFN with residual connections |
| Final LayerNorm | Applied to the last block's output before the LM head |
| LM Head         | Final linear layer projecting to vocabulary logits, with loss computed in-model when targets are given |

### Default hyperparameters (`config.py`)

| Parameter       | Value |
|------------------|-------|
| Embedding dimension | 64 |
| Block size (context length) | 128 |
| Number of layers | 4 |
| Number of attention heads | 4 |
| Batch size | 32 |
| Epochs | 5 |
| Training steps per chunk | 200 |
| Learning rate | 3e-4 (cosine annealed) |
| Eval steps | 20 |
| Chunk size (token budget) | 2,000,000 tokens |
| Dropout | 0.1 |
| Tokenizer | `o200k_base` (tiktoken) |

## Dataset

The model is trained on [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories), a dataset of short, simple children's stories — well suited for training small language models from scratch with limited compute.

Text is tokenized with `tiktoken`'s `o200k_base` BPE encoding. Stories are split 95/5 into train and validation sets before tokenization, so validation data never leaks into the training chunks. Each story is encoded and terminated with an end-of-text token, and the training portion is saved as chunks sized by a fixed token budget (rather than story count), keeping chunk size — and therefore memory and load time — predictable.

## Setup

This project was built to run on **Google Colab** with data stored on Google Drive.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ashgpt.git
cd ashgpt
```

### 2. Install dependencies

```bash
pip install torch datasets tiktoken
```

### 3. Configure storage

`preprocessor.py` mounts Google Drive and defines `save_dir`, where vocab config, data chunks, and checkpoints are stored. Update `save_dir` if you want to use local storage instead of Google Drive, or if you're not running in Colab.

## Usage

### 1. Preprocess the data

Downloads the dataset, splits it into train/validation sets, tokenizes everything with the BPE tokenizer, and saves it as token-budgeted chunks:

```bash
python preprocessor.py
```

This produces, inside `save_dir`:
- `config.pt` — dataset config (encoding name, chunk count, train/val sizes, dataset name)
- `chunk_0.pt`, `chunk_1.pt`, … — tokenized training data chunks (int32)
- `val_chunk.pt` — held-out validation data

### 2. Train the model

```bash
python train.py
```

While one chunk trains, the next is prefetched in the background so the GPU doesn't wait on disk I/O. Training runs a fixed number of steps per chunk per epoch, with gradient clipping, mixed precision, and a cosine-annealed learning rate, and evaluates on the validation set at the end of each epoch. The full training state is checkpointed whenever validation loss improves.

### 3. Generate text

```python
from generate import generate

print(generate("Once upon a time", max_token=100, temparature=0.8, top_k=50))
```

`generate()` loads the best checkpoint, encodes the prompt, and samples a continuation using temperature scaling and top-k filtering, stopping early if it produces the end-of-text token.

## Repository Info

**Name:** `ashgpt`

**Description:** AshGPT — a GPT-style language model built from scratch in PyTorch, with BPE tokenization, a properly regularized and normalized transformer architecture, async data prefetching, mixed-precision training, and controllable text generation.
