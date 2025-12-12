# RAG_LLM_project

📚 Retrieval-Augmented Generation (RAG) project — course project for NLP.

This repository contains a RAG system that combines information retrieval with Large Language Models (LLMs) to produce accurate, context-aware answers grounded in external data.

## Project overview

The project focuses on:

- Implementing a RAG pipeline (Retriever + LLM)
- Experimenting with different retrieval strategies (e.g., BM25, embedding-based retrieval)
- Testing and comparing LLM behavior under different prompting and retrieval setups
- Building a modular and extensible experimental environment for RAG-based QA systems

RAG retrieves relevant documents or text chunks and passes them as context to an LLM, which reduces hallucinations and improves answer quality.

## Important note for reviewers

Please focus your review on the `new_cursor_local` branch — it contains:

- The most up-to-date implementation
- Current architectural decisions
- Integrated retrieval + LLM logic
- Latest experiments and improvements

Other branches are experimental or exploratory.

## Branch structure

- main  
  The base and stable branch containing the initial structure, notebooks, and general project setup.

- new_cursor_local (PRIMARY BRANCH FOR REVIEW)  
  Main development branch and the most representative version for evaluation. Includes:
  - The latest RAG pipeline implementation
  - Improved local context/index management
  - Refined retrieval logic
  - Integration with LLMs
  - Updated experiments and utilities

- testy_LLM  
  Dedicated to LLM testing: prompt experiments, sanity checks, and comparisons of prompting strategies.

- experiment/better-llm  
  Experimental ideas aimed at improving LLM performance (alternate prompts, model configurations, etc.).

- BM25-Retriever  
  BM25-based retriever implementation for comparison with embedding-based approaches.

- File-Search-Tool-Google  
  Experimental work on file search and indexing tools inspired by Google-style search and document exploration.

## How the RAG pipeline works

1. Data ingestion (documents, text files, notebooks, etc.)
2. Indexing / retrieval setup (BM25 or embeddings)
3. User query
4. Relevant context retrieval
5. Context + query passed to the LLM
6. LLM generates a grounded, context-aware answer

This lets the model answer based on real data rather than relying solely on parametric knowledge.

## Example setup (general)

1. Clone the repository
   git clone https://github.com/Wojz12/RAG_LLM_project.git

2. Switch to the main development branch
   git checkout new_cursor_local

3. Install dependencies
   pip install -r requirements.txt

4. Run tests (example)
   pytest

5. Run a sample RAG query (example)
   python run_rag.py --query "Your question here"

(Commands may vary depending on the current scripts and structure in the branch.)

## Testing

- LLM-focused tests are mainly in the `testy_LLM` branch.
- Integrated retrieval + generation tests and experiments are present in `new_cursor_local`.

Testing ensures correctness of the RAG pipeline, reasonable answer quality, and robustness to different queries and edge cases.

## Evaluation focus

When reviewing the project, consider:

- Design of the RAG pipeline
- Retrieval → LLM integration
- Clarity and modularity of the code
- Quality and structure of LLM tests
- Experimental depth and reasoning behind design choices

## Contributing

Contributions, bug reports, and suggestions are welcome. Please open issues or pull requests against the `new_cursor_local` branch for development-related changes.
