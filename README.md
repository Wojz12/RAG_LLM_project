# RAG_LLM_project
📚 RAG_LLM_project

This repository contains a Retrieval-Augmented Generation (RAG) project built as part of an NLP course project.
The goal is to design and experiment with a system that combines information retrieval with Large Language Models (LLMs) to generate accurate, context-aware answers based on external data.

🚀 Project Overview

The project focuses on:

implementing a RAG pipeline (Retriever + LLM),

experimenting with different retrieval strategies (e.g. BM25, embeddings),

testing and comparing LLM behavior under different prompting and retrieval setups,

building a modular and extensible experimental environment for RAG-based QA systems.

RAG works by retrieving relevant documents or text chunks first, and then passing them as context to the LLM, significantly reducing hallucinations and improving answer quality.

⭐ Important Note for Reviewers

👉 Please focus mainly on the new_cursor_local branch, as it contains:

the most up-to-date implementation,

the latest architectural decisions,

integrated retrieval + LLM logic,

current experiments and improvements,

the most representative version of the project.

Other branches are experimental or exploratory.

🌳 Branch Structure
🔹 main

The base and stable branch of the repository.
Contains the initial structure, notebooks, and general project setup.

⭐ new_cursor_local (PRIMARY BRANCH FOR REVIEW)

This is the main development branch and the most important one for evaluation.

It includes:

the latest RAG pipeline implementation,

improved local context/index management,

refined retrieval logic,

integration with LLMs,

updated experiments and utilities,

code prepared for testing and further experimentation.

✅ This branch best represents the final state of the project.

🔹 testy_LLM

Dedicated to LLM testing:

prompt experiments,

quality checks of generated answers,

functional and sanity tests,

comparison of different prompting strategies.

This branch focuses on evaluating how the LLM behaves under different conditions.

🔹 experiment/better-llm

Contains experimental ideas aimed at improving LLM performance, such as:

alternative prompting strategies,

different model configurations,

exploratory improvements to generation quality.

🔹 BM25-Retriever

Implements and tests a BM25-based retriever:

classical keyword-based retrieval,

used for comparison with embedding-based approaches,

helpful for benchmarking retrieval quality.

🔹 File-Search-Tool-Google

Experimental branch for file search and indexing tools, inspired by Google-style search mechanisms and document exploration.

🧠 How the RAG Pipeline Works

Data ingestion (documents, text files, notebooks, etc.)

Indexing / retrieval setup (BM25 or embeddings)

User query

Relevant context retrieval

Context + query passed to the LLM

LLM generates a grounded, context-aware answer

This approach allows the model to base its answers on real data instead of relying only on parametric knowledge.

🧪 Testing

LLM-focused tests are located mainly in the testy_LLM branch.

Integrated retrieval + generation tests and experiments are present in new_cursor_local.

Testing ensures:

correctness of the RAG pipeline,

reasonable answer quality,

robustness to different queries and edge cases.

📦 Example Setup (General)
# Clone the repository
git clone https://github.com/Wojz12/RAG_LLM_project.git

# Switch to the main development branch
git checkout new_cursor_local

# Install dependencies
pip install -r requirements.txt

# Run tests (example)
pytest

# Run a sample RAG query
python run_rag.py --query "Your question here"


(Commands may vary depending on the current scripts and structure in the branch.)

🎯 Evaluation Focus

When reviewing the project, please consider:

the design of the RAG pipeline,

retrieval → LLM integration,

clarity and modularity of the code,

quality and structure of LLM tests,

experimental depth and reasoning behind design choices.
