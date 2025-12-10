# Quran Guider using RAG

![RAG Workflow](RAG_workflow.jpg)

A beginner‑friendly Retrieval‑Augmented Generation (RAG) project that answers questions about the Qur’an using a PDF of **“The Qur’an with Annotated Interpretation in Modern English – Ali Ünal”**, dense embeddings, FAISS as a vector database, and the **Gemma‑2‑2B‑IT** language model.

Everything is implemented in a single Colab‑ready notebook:

> `QuranGuiderUsingRAG.ipynb`

---

## 🌟 Features

- Load and parse the Qur’an PDF page by page.
- Clean the text (remove inline numeric citations like `).22`) and split it into sentences.
- Create fixed‑size chunks of **10 sentences** (last chunk may be shorter).
- Generate dense embeddings for every chunk using **sentence-transformers/all-mpnet-base-v2**.
- Store embeddings in a **FAISS** index for fast similarity search.
- Load **Gemma‑2‑2B‑IT** in 4‑bit with `bitsandbytes` and Hugging Face Transformers.
- Answer English questions about the Qur’an by:
  1. Encoding the question.
  2. Retrieving the top‑8 most relevant chunks from FAISS.
  3. Building a prompt that includes the retrieved context.
  4. Letting Gemma generate a grounded answer.

---

## 📁 Project structure

├── QuranGuiderUsingRAG.ipynb # Main notebook (all code)
├── RAG_workflow.jpg # RAG workflow diagram (used in README)
└── the-quran-with-annotated-interpretation-in-modern-english-ali-unal.pdf

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/SyedNajiullah/QuranGuiderUsingRAG.git
cd QuranGuiderUsingRAG
```
### 2. Add the Qur’an PDF (optional)

Add the pdf to root as the-quran-with-annotated-interpretation-in-modern-english-ali-unal.pdf

### 3. Install dependencies

Run the cell one by one it will automatically install all dependencies. You might need to restart the session once when the google gemma is loaded for the bitsandbytes error if encountered.


### 4. Hugging Face token (for Gemma)

Gemma is a **gated** model. To use `google/gemma-2-2b-it`:

1. Create a Hugging Face account and log in.  
2. Open the model page and accept the license:  
   - https://huggingface.co/google/gemma-2-2b-it  
3. Go to your tokens page:  
   - https://huggingface.co/settings/tokens  
4. Create a **fine‑grained token** with:
   - **Read** access to models.
   - Permission to access **public gated models**.
5. In your notebook, set:
   - HF_TOKEN = "hf_YourHuggingFaceToken"


> Never commit your token to GitHub.

---

## 🧱 How it works (high‑level)

### 1. PDF loading and chunking

In `QuranGuiderUsingRAG.ipynb`:

- The PDF is opened with `pdfplumber`.
- For each page:
  - Text is extracted.
  - Citation‑style numbers right after `.` or `)` are removed via regex.
  - Text is split into sentences using `nltk.sent_tokenize`.
  - Sentences are grouped into chunks of **10 sentences**.

Each chunk is stored in a pandas DataFrame with at least:

- `page_number`  
- `chunk_id_in_page`  
- `chunks_text`  

### 2. Embedding generation and FAISS index

The notebook uses **all-mpnet-base-v2** from Sentence-Transformers and stores the embeddings in FAISS (Facebook Artificial Intelligence Similarity Search) vector database.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(
"sentence-transformers/all-mpnet-base-v2",
device=device,
)

chunk_texts = chunks_df["chunks_text"].tolist()
emb_matrix = embedding_model.encode(chunk_texts, convert_to_numpy=True).astype("float32")

faiss.normalize_L2(emb_matrix)

d = emb_matrix.shape​
index = faiss.IndexFlatIP(d)
index.add(emb_matrix)

print("Index size:", index.ntotal)
```

### 3. Loading Gemma‑2‑2B‑IT in 4‑bit

During this project I faced the issue of Low RAM. Initially I wanted to use Gemma-7B-it model but it required more then 28GB VRAM for FLoat32. Therefore I had do use Gemma‑2‑2B‑IT in 4‑bit from hugging face.
```python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


HF_TOKEN = "hy_token_goes_here"

model_id = "google/gemma-2-2b-it"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    quantization_config=quant_config,
    device_map="auto",          
    low_cpu_mem_usage=True,     
)

```
