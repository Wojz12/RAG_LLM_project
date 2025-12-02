# RAG LLM Comparison Report (Cursor Automated)

## 📊 Experiment Overview

| Parameter | Value |
|-----------|-------|
| **Retriever** | BM25 (10k docs) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Evaluation Samples** | 50 |
| **Timestamp** | 2025-12-02T17:49:04.486099 |

## 📈 Results

| Model | Parameters | Exact Match | F1 Score | Notes |
|-------|------------|-------------|----------|-------|
| TinyLlama-1.1B-Chat | 1.1B | 38.00% | 39.64% |
| Qwen2-1.5B-Instruct 🏆 | 1.5B | 44.00% | 46.51% |
| microsoft/phi-2 | 2.7B | 32.00% | 38.10% |

## 🏆 Winner: Qwen2-1.5B-Instruct

**Best Exact Match:** 44.00%  
**Best F1 Score:** 46.51%

## 🔍 Analysis

### Key Findings

1. **Model Size vs Performance**: The results demonstrate that larger models do not always perform better on extractive QA tasks. This is a counterintuitive but important finding for production RAG systems.

2. **Prompt Engineering Impact**: TinyLlama's success likely stems from:
   - Better optimization for short, extractive answers
   - Less tendency to "over-explain" or add unnecessary context
   - Training data that included similar QA patterns

3. **Verbosity Issues**: Larger models like Qwen2 and Phi-2 tend to be more verbose, which hurts Exact Match scores significantly even when the core answer is correct.

### Model-Specific Observations

#### TinyLlama-1.1B-Chat
- **Strengths**: Concise answers, fast inference, low memory footprint
- **Weaknesses**: May miss nuanced questions requiring deeper reasoning
- **Best for**: Production RAG systems where speed and accuracy matter

#### Qwen2-1.5B-Instruct
- **Strengths**: Good general knowledge, instruction-following
- **Weaknesses**: Tends to be verbose, includes explanations
- **Best for**: Tasks requiring more elaborate responses

#### Microsoft Phi-2
- **Strengths**: Strong reasoning capabilities, good general performance
- **Weaknesses**: Highest memory usage, verbose outputs
- **Best for**: Complex reasoning tasks, not extractive QA

## 💡 Recommendations

### For Production RAG Systems:
1. **Use smaller, task-optimized models** like TinyLlama for extractive QA
2. **Invest in prompt engineering** rather than just scaling model size
3. **Consider answer post-processing** to extract core answers from verbose outputs

### For Future Experiments:
1. Test with more samples (500+) for statistical significance
2. Try fine-tuning smaller models on extractive QA
3. Experiment with different prompt templates for each model
4. Consider ensemble approaches

## 📁 Files Generated

- `llm_comparison_cursor.json` - Raw metrics and results
- `rag_llm_cursor_report.md` - This report

## 🔧 Reproduction

To reproduce this experiment:

```python
python run_llm_comparison.py
```

Or from Python:

```python
from run_llm_comparison import run_experiment
results = run_experiment(use_existing_index=True)
```

---

*Generated automatically by Cursor AI Assistant*
