# Open-Domain QA with RAG (TriviaQA)

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for Open-Domain Question Answering using the **TriviaQA** dataset.

**Architecture:** BM25 Retrieval → CrossEncoder Reranking → TinyLlama Generation

## 🏆 Results

### Main Results (500 samples)
| Model | Index Size | Eval Samples | Exact Match | F1 Score |
|-------|------------|--------------|-------------|----------|
| BM25 + Reranker + TinyLlama | 10,000 | 500 | **39.80%** | **44.36%** |

*Evaluated on TriviaQA validation set (rc.wikipedia configuration)*

### 🔬 LLM Comparison Experiment (500 samples)

We tested different LLMs to find the best generator for our RAG pipeline:

| Model | Parameters | Exact Match | F1 Score | Notes |
|-------|------------|-------------|----------|-------|
| **TinyLlama-1.1B-Chat** 🏆 | 1.1B | **39.80%** | **44.36%** | Winner - best prompt optimization |
| Qwen2-1.5B-Instruct | 1.5B | 24.00% | 28.80% | Too verbose |
| Microsoft Phi-2 | 2.7B | 18.00% | 28.04% | Larger but worse on extractive QA |

**Key Finding:** Bigger model ≠ better results! TinyLlama with task-specific prompt engineering outperforms larger models on extractive QA tasks.

## 🛠️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │ ──► │  BM25       │ ──► │  Reranker   │ ──► │  TinyLlama  │
│             │     │  Retriever  │     │ CrossEncoder│     │  Generator  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                         │                    │                    │
                    Top-10 docs          Top-3 docs            Answer
```

### Components:

| Component | Implementation | Description |
|-----------|----------------|-------------|
| **Retriever** | BM25 (rank_bm25) | Sparse keyword-based retrieval |
| **Reranker** | CrossEncoder (ms-marco) | Semantic reranking of candidates |
| **Generator** | TinyLlama-1.1B-Chat | Answer generation from context |

## 📂 Dataset

**Source:** [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (rc.wikipedia)

### Data Splits (per assignment requirements):
| Split | Source | Size |
|-------|--------|------|
| **Validation** | First 7,900 from train | 7,900 |
| **Train** | Rest of train (7,900+) | ~70,000 |
| **Test** | Original validation | 7,993 |

## 🚀 Quick Start (Google Colab)

### 1. Setup
```python
!pip install -q torch transformers datasets accelerate rank_bm25 sentence-transformers tqdm

!git clone -b new_cursor_local https://github.com/Wojz12/RAG_LLM_project.git
%cd RAG_LLM_project
```

### 2. Load Data & Build Index
```python
from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever
from src.utils import prepare_corpus

data = load_trivia_qa()
train_data = data["train"].select(range(7900, 17900))  # 10k samples

corpus = prepare_corpus(train_data)
sparse = SparseRetriever()
sparse.build_index(corpus)
```

### 3. Load Models
```python
from src.re_ranker import Reranker
from src.generator import RAGGenerator

reranker = Reranker()
generator = RAGGenerator()
```

### 4. Ask Questions
```python
query = "Who wrote Romeo and Juliet?"

docs = sparse.retrieve(query, top_k=10)
contexts = reranker.rerank(query, [d["text"] for d in docs], top_k=3)
answer = generator.generate_answer(query, "\n\n".join(contexts))

print(f"Answer: {answer}")
```

### 5. Evaluate on Test Set
```python
from src.evaluation import evaluate_predictions
from tqdm import tqdm

EVAL_SIZE = 500
eval_data = test_data.select(range(EVAL_SIZE))

predictions = {}
references = {}

for example in tqdm(eval_data, desc="Evaluating"):
    q = example["question"]
    q_id = example["question_id"]
    
    docs = sparse.retrieve(q, top_k=10)
    texts = [d["text"] for d in docs]
    contexts = reranker.rerank(q, texts, top_k=3)
    answer = generator.generate_answer(q, "\n\n".join(contexts))
    
    predictions[q_id] = answer
    references[q_id] = example["answer"]["aliases"]

metrics = evaluate_predictions(predictions, references)
print(f"Exact Match: {metrics['exact_match']:.2f}%")
print(f"F1 Score: {metrics['f1']:.2f}%")
```

## 🔬 LLM Comparison (Google Colab)

Code to compare different LLMs for the RAG pipeline:

### Test Qwen2-1.5B-Instruct
```python
from src.generator import RAGGenerator
from src.evaluation import evaluate_predictions
from tqdm import tqdm

# Load Qwen2 model
generator_qwen = RAGGenerator(model_name="Qwen/Qwen2-1.5B-Instruct")

# Evaluate
predictions_qwen = {}
references_qwen = {}

for example in tqdm(eval_data, desc="Qwen2-1.5B"):
    q = example["question"]
    q_id = example["question_id"]
    
    docs = sparse.retrieve(q, top_k=10)
    texts = [d["text"] for d in docs]
    contexts = reranker.rerank(q, texts, top_k=3)
    answer = generator_qwen.generate_answer(q, "\n\n".join(contexts))
    
    predictions_qwen[q_id] = answer
    references_qwen[q_id] = example["answer"]["aliases"]

metrics_qwen = evaluate_predictions(predictions_qwen, references_qwen)
print(f"Qwen2: EM={metrics_qwen['exact_match']:.2f}%, F1={metrics_qwen['f1']:.2f}%")
```

### Test Microsoft Phi-2 (2.7B)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc

# Cleanup previous models
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

MODEL_NAME = "microsoft/phi-2"

tokenizer_phi = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_phi = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"

if tokenizer_phi.pad_token is None:
    tokenizer_phi.pad_token = tokenizer_phi.eos_token

def generate_phi2(query: str, context: str) -> str:
    context = context[:1500]
    prompt = f"""Answer the question based on the context. Give a short answer (1-5 words only).

Context: {context}

Question: {query}
Answer:"""
    
    inputs = tokenizer_phi(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
    
    with torch.no_grad():
        outputs = model_phi.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer_phi.pad_token_id,
        )
    
    response = tokenizer_phi.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    answer = response.strip().split('\n')[0].split('Question:')[0]
    if '.' in answer:
        answer = answer.split('.')[0]
    return answer.strip()

# Evaluate Phi-2
predictions_phi = {}
references_phi = {}

for example in tqdm(eval_data, desc="Phi-2"):
    q = example["question"]
    q_id = example["question_id"]
    
    docs = sparse.retrieve(q, top_k=10)
    texts = [d["text"] for d in docs]
    contexts = reranker.rerank(q, texts, top_k=3)
    answer = generate_phi2(q, "\n\n".join(contexts))
    
    predictions_phi[q_id] = answer
    references_phi[q_id] = example["answer"]["aliases"]

metrics_phi = evaluate_predictions(predictions_phi, references_phi)
print(f"Phi-2: EM={metrics_phi['exact_match']:.2f}%, F1={metrics_phi['f1']:.2f}%")
```

### Save Comparison Results
```python
import json

comparison_results = {
    "experiment": "LLM Comparison for RAG QA",
    "retriever": "BM25 (10k docs)",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "eval_samples": 500,
    "models": [
        {"name": "TinyLlama-1.1B-Chat", "params": "1.1B", "em": 39.80, "f1": 44.36, "winner": True},
        {"name": "Qwen2-1.5B-Instruct", "params": "1.5B", "em": 24.00, "f1": 28.80, "winner": False},
        {"name": "microsoft/phi-2", "params": "2.7B", "em": 18.00, "f1": 28.04, "winner": False},
    ],
    "conclusion": "Smaller task-optimized model outperforms larger general models"
}

with open("llm_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)
print("Results saved!")
```

## 📁 Project Structure

```
RAG_LLM_project/
├── src/
│   ├── data_loader.py    # TriviaQA data loading
│   ├── retriever.py      # BM25 & Dense retrievers
│   ├── re_ranker.py      # CrossEncoder reranking
│   ├── generator.py      # TinyLlama generation
│   ├── evaluation.py     # EM & F1 metrics
│   ├── utils.py          # Corpus preparation
│   └── pipeline.py       # Full pipeline
├── colab_hybrid_rag.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Local Setup

```bash
# Create environment
conda create -n rag-project python=3.10
conda activate rag-project

# Install dependencies
pip install -r requirements.txt
```

## 📊 Evaluation Metrics

- **Exact Match (EM):** Percentage of predictions that exactly match ground truth
- **F1 Score:** Token-level overlap between prediction and ground truth

## 📜 References

- Dataset: [TriviaQA (Joshi et al., 2017)](https://aclanthology.org/P17-1147/)
- Reranker: [MS MARCO CrossEncoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- Generator: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## 👨‍💻 Author

University Final Project - Open-Domain Question Answering with RAG
