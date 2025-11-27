# Open-Domain QA with RAG (TriviaQA)

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for Open-Domain Question Answering using the **TriviaQA** dataset.

**Architecture:** BM25 Retrieval → CrossEncoder Reranking → TinyLlama Generation

## 🏆 Results

| Model | Index Size | Eval Samples | Exact Match | F1 Score |
|-------|------------|--------------|-------------|----------|
| BM25 + Reranker + TinyLlama | 10,000 | 500 | **39.80%** | **44.36%** |

*Evaluated on TriviaQA validation set (rc.wikipedia configuration)*

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
