# Open-Domain QA with RAG (TriviaQA)

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for Open-Domain Question Answering. It is designed to answer factoid questions using the **TriviaQA** dataset by retrieving evidence from Wikipedia and generating answers using a Large Language Model (LLM).

**Current Phase:** Phase 1 (Sparse Retrieval Baseline)

## 📂 Dataset & Constraints
We use the **TriviaQA** dataset (RC configuration).
* **Source:** [HuggingFace Viewer](https://huggingface.co/datasets/mandarjoshi/trivia_qa/viewer/rc/train)
* **Configuration:** `rc.wikipedia` (Focusing on Wikipedia evidence first as recommended).

### 🚨 Critical Data Split
[cite_start]Per assignment requirements[cite: 72, 73], we do **not** use the default validation set for validation.
1.  **Training Set:** `train` split (indices 7,900 to end).
2.  **Validation Set:** The **first 7,900 examples** of the `train` split.
3.  **Test Set:** The original `validation` split.

## 🛠️ Architecture

### 1. Retrieval (Sparse First)
* **Method:** BM25 (Best Matching 25).
* **Library:** `rank_bm25`.
* **Logic:**
    * Extracts `EntityPages` (Wikipedia) from the dataset.
    * Chunks long documents into smaller passages (e.g., 256 tokens).
    * Indexes passages for keyword-based retrieval.

### 2. Generation (To Be Implemented)
* **Model:** Quantized LLM (e.g., Qwen/Llama-3 via `bitsandbytes`).
* **Method:** 4-bit quantization to fit consumer GPUs.

## ⚙️ Setup

1.  **Create Environment:**
    ```bash
    conda create -n rag-project python=3.10
    conda activate rag-project
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Requirements (`requirements.txt`):**
    ```text
    torch
    transformers
    datasets
    accelerate
    bitsandbytes
    rank_bm25
    scikit-learn
    tqdm
    # faiss-gpu (Reserved for Phase 2: Dense Retrieval)
    ```

## 🚀 Usage

### 1. Data Loading & Splitting
The data loader automatically handles the 7,900-split rule:
```python
from src.data_loader import load_trivia_qa
data = load_trivia_qa(split="rc.wikipedia")
# Returns: {'train': ..., 'val': (first 7.9k), 'test': ...}
2. Running the Sparse Retriever
Bash

python src/pipeline.py --mode sparse --query "What film did Marie Curie inspire?"
3. Evaluation
Evaluates using Exact Match (EM) and F1 scores using the official TriviaQA metrics.

Bash

python src/evaluation.py --predictions output/preds.json

## Running in Google Colab (Hybrid RAG)

To run the full Hybrid RAG pipeline (BM25 + Dense Retrieval + LLM Generation) on Google Colab with GPU acceleration:

1. **Open the Colab Notebook:**
   - Upload `colab_hybrid_rag.ipynb` to Google Colab or use the direct link.

2. **Enable GPU Runtime:**
   - Go to `Runtime` → `Change runtime type` → Select `T4 GPU` (or higher).

3. **Install Dependencies:**
   ```python
   !pip install -q datasets rank_bm25 transformers accelerate bitsandbytes faiss-gpu
   ```

4. **Run the Pipeline:**
   ```python
   from src.pipeline import run_hybrid_pipeline
   results = run_hybrid_pipeline(questions, top_k=5)
   ```

5. **Memory Tips for Colab Free Tier:**
   - Use `load_in_4bit=True` for quantized LLM inference.
   - Process questions in batches to avoid OOM errors.
   - Clear GPU cache between runs: `torch.cuda.empty_cache()`.

📈 Git Hygiene (Grading Requirement)
Commit often: Do not squash changes.

History: Reflects incremental progress (Data Loader -> Sparse Retriever -> Generator -> Dense Retrieval).

📜 References

Dataset: TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension (Joshi et al., 2017) 



Assignment: Final Project RAG Specifications.


---

### Suggested Git Commit
Since we are establishing the project root, you should commit this now.

**Command:**
```bash
git add README.md
git commit -m "Docs: Initialize project README with TriviaQA constraints and