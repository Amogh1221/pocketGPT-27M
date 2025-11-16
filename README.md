# pocketGPT: A Custom 27M Parameter GPT Model Trained From Scratch

This repository documents the full pipeline for creating **pocketGPT**, a compact ~27M parameter GPT-style model that is **pretrained from scratch** on a Machine Learning corpus and later **finetuned** on a structured dialog dataset.

This project demonstrates end‑to‑end LLM development on consumer‑level or Kaggle‑level hardware, including:

* Custom tokenizer training
* GPT‑2 style architecture design
* Pretraining from raw text
* Finetuning for conversational ML assistance
* Exporting and running the model locally

---

## Key Features

* **Custom Byte‑Level BPE tokenizer** (vocab size: 24k)
* **27M parameter GPT‑2 architecture** optimized for low‑compute training
* **Context window:** 384 tokens
* Full **pretraining + finetuning** pipeline included
* Works efficiently on **Kaggle T4 GPUs (12‑hour limit)**
* BF16 / mixed‑precision support
* Clean and reproducible training workflow

---

## Dataset

The model is trained in two phases:

### 1. Pretraining

* Dataset: `ML_corpus.txt`
* Size: ~165M tokens (after tokenization)
* Task: Causal LM (standard GPT pretraining)

A 99/1 split is applied:

```
train: 99%
test: 1%
```

### 2. Finetuning

* Dataset: Dialog pairs CSV
* ~139k rows
* ~2.6M tokens total
* Task: SFT (supervised instruction tuning)

This makes the model behave like a compact ML doubt‑solving assistant.

---

## Tokenizer

A custom Byte‑Level BPE tokenizer is trained:

```
vocab_size = 24000
special_tokens = ["<pad>", "<unk>"]
```

Reasons for a custom tokenizer:

* Faster training
* Smaller model size
* Better compression for ML terms

---

## Model Architecture

pocketGPT uses a compact GPT‑2 style configuration:

```
n_embd = 384
n_layer = 8–10
n_head = 6–7
context_window = 384
total_params ≈ 27M
```

This provides a stable balance between:

* Training speed
* Model quality
* Compute requirements

---

## Training Details

### Pretraining

* BF16 mixed precision
* Gradient accumulation to simulate larger batch sizes
* Warmup + cosine‑style adjustments
* Logging every 500 steps

### Finetuning

* Same tokenizer and model weights
* Small‑batch conversational SFT
* Produces short‑form Q/A‑style answers similar to dataset

---

## Output Files

The final exported model directory contains:

```
pocketGPT/
 ├── config.json
 ├── pytorch_model.bin
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── vocab.json
 ├── merges.txt
 ├── special_tokens_map.json
```

You can download this zip for local usage or directly use with this example given below.

---

## Inference Example

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import os

os.environ["HUGGINGFACE_HUB_TOKEN"] = "Your Tokens"

model = GPT2LMHeadModel.from_pretrained("Amogh1221/PocketGPT_27M")
tokenizer = GPT2TokenizerFast.from_pretrained("Amogh1221/PocketGPT_27M")

def ask(prompt):
    formatted = f"<|bos|>Instruction: {prompt}\nResponse:"
    
    inputs = tokenizer.encode(formatted, return_tensors="pt")
    inputs = inputs.to(model.device) 

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=384,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

ask("what is an Artificial Neural Network?")

```

---

## Limitations

* Cannot match reasoning depth of large LLMs
* Shorter answers due to smaller context window
* Knowledge limited strictly to training corpus

---

## Future Improvements

* Larger context window (512–1024)
* Multi‑epoch SFT on higher‑quality datasets
* RLHF (reward‑based tuning)
* GGUF export for local CPU inference

---

## License

MIT License
