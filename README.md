# Retrieval-Augmented Generation (RAG) with TriviaQA (Wikipedia Evidence)

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline for question answering. The system combines a sparse retriever (BM25) with a powerful language model (FLAN-T5) to answer questions based on information retrieved from a given corpus.

## Project Components

### 1. Data Loading and Preparation

- **Dataset**: The project utilizes the `trivia_qa` dataset, specifically the `rc.wikipedia` configuration, which contains questions and corresponding Wikipedia passages as evidence. This dataset is loaded using the `datasets` library.
- **Corpus Creation**: A text corpus is built from the `entity_pages` field of the TriviaQA dataset. This field often contains `wiki_context` or `wiki_page` with relevant Wikipedia articles. To handle cases where the context might be very long, the text is chunked into smaller, overlapping segments (`max_len=256`, `overlap=50`). This ensures that individual documents passed to the retriever are manageable in size.

### 2. BM25 Retriever

- **Purpose**: The BM25 retriever is responsible for efficiently searching the corpus and identifying the most relevant documents for a given question.
- **Algorithm**: It uses the BM25Okapi algorithm, a widely used ranking function that estimates the relevance of documents to a given search query. It's a sparse retrieval method, meaning it relies on word overlap and term frequency-inverse document frequency (TF-IDF) principles.
- **Implementation**: The corpus documents are tokenized (split by whitespace), and an `BM25Okapi` object is initialized with the tokenized corpus. When a query is provided, it's also tokenized, and the `get_scores` method returns a relevance score for each document in the corpus. The top-k documents with the highest scores are then selected as contexts.

### 3. FLAN-T5 Generator

- **Purpose**: The FLAN-T5 generator is a large language model that takes the retrieved documents (contexts) and the original question as input, then generates a coherent and concise answer.
- **Model**: We use `google/flan-t5-base`, an instruction-tuned sequence-to-sequence model known for its ability to follow instructions and generate high-quality text.
- **Efficiency**: To ensure the model can run effectively even on resource-constrained environments like a free Colab GPU, 4-bit quantization is applied during model loading. This significantly reduces memory usage while maintaining performance.
- **Prompt Engineering**: The question and retrieved contexts are formatted into a specific prompt structure: `"Answer the question using the provided context.\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"`. This clearly instructs the model to use the provided information to generate its answer.

## How the RAG Pipeline Works

1. **Question Input**: The user provides a question.
2. **Retrieval**: The BM25 retriever takes the question, tokenizes it, and queries the pre-indexed corpus. It returns the top-k most relevant text chunks (contexts).
3. **Generation**: These retrieved contexts, along with the original question, are fed into the FLAN-T5 language model as part of a carefully constructed prompt.
4. **Answer Output**: The FLAN-T5 model generates an answer based *only* on the information present in the provided contexts. This helps to reduce hallucinations and ensure factual consistency.

## Setup and Usage

### Prerequisites

- Python 3.x
- Google Colab (recommended for GPU access)

### Installation

```bash
!pip install datasets
!pip install rank-bm25
!pip install transformers accelerate bitsandbytesRunning the Pipeline
The notebook demonstrates the following steps:

Load Dataset: from datasets import load_dataset and dataset = load_dataset("trivia_qa", "rc.wikipedia")
Inspect Data: Print examples from the dataset splits (train, validation, test) to understand its structure.
Build Corpus: The provided code iterates through a subset of the training data (e.g., 3000 examples for quick demonstration) to extract wiki_context from entity_pages and chunks them into smaller pieces. This processed list of text chunks becomes corpus.
Initialize BM25 Retriever: The corpus is tokenized, and BM25Okapi is initialized: bm25 = BM25Okapi(tokenized_corpus).
Load FLAN-T5 Generator: The google/flan-t5-base model and its tokenizer are loaded, with 4-bit quantization for efficiency.
Define generate_answer Function: This function takes a question and a list of contexts, constructs the prompt, and uses the FLAN-T5 model to generate an answer.
Define retrieve_bm25 Function: This helper function wraps the BM25 retrieval logic, returning the top-k documents for a given query.
Define rag_pipeline Function: This function combines the retrieve_bm25 and generate_answer functions to provide an end-to-end RAG solution.
Example Usage: A sample question is used to demonstrate the full RAG pipeline, printing both the question and the model's generated answer.
Example Output
When running the example query, you would typically see output similar to:

🧠 Question: From which country did Angola achieve independence in 1975?

💬 Model answer: Angola
(Note: The corpus and tokenized_corpus variables were re-evaluated due to the chunking implementation, which might lead to slightly different output from the initial quick test in the notebook.)

This project provides a solid foundation for building more advanced RAG systems, which can be further improved by exploring different retrieval methods (e.g., dense retrievers), more sophisticated chunking strategies, and fine-tuning the generator model.
