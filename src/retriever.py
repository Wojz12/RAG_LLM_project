import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import pickle
import os
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """
    Abstract Base Class for all Retrievers (Sparse & Dense).
    """
    
    @abstractmethod
    def build_index(self, corpus: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def save_index(self, path: str):
        pass
    
    @abstractmethod
    def load_index(self, path: str):
        pass


class SparseRetriever(BaseRetriever):
    """
    BM25-based Sparse Retriever using rank_bm25.
    """
    
    def __init__(self):
        self.bm25 = None
        self.corpus = [] # Store full corpus to retrieve text later
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Improved tokenization:
        - Lowercase
        - Remove punctuation
        - Split by whitespace
        """
        text = text.lower()
        # Replace punctuation with space to preserve words
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def build_index(self, corpus: List[Dict[str, Any]]):
        """
        Builds BM25 index from a list of documents.
        """
        logger.info(f"Building BM25 index for {len(corpus)} passages...")
        self.corpus = corpus
        
        # Tokenize corpus
        # Use a list comprehension with progress bar
        tokenized_corpus = []
        for doc in tqdm(corpus, desc="Tokenizing corpus"):
             tokenized_corpus.append(self._tokenize(doc["text"]))
        
        # Build BM25
        logger.info("Initializing BM25Okapi (this may take a while for large corpora)...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("Index not built! Call build_index() or load_index() first.")
        
        tokenized_query = self._tokenize(query)
        
        # Get top_k scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Optimize sorting: use argpartition for large arrays instead of full sort
        # We only need top_k
        if len(scores) > top_k:
            top_n_indices = np.argpartition(scores, -top_k)[-top_k:]
            # The top_k are not sorted within themselves, so sort them now
            top_n_indices = top_n_indices[np.argsort(scores[top_n_indices])][::-1]
        else:
            top_n_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_n_indices:
            doc = self.corpus[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
            
        return results

    def save_index(self, path: str):
        """
        Saves both the BM25 object and the corpus to a pickle file.
        """
        logger.info(f"Saving index to {path}...")
        data = {
            "bm25": self.bm25,
            "corpus": self.corpus
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Index saved.")

    def load_index(self, path: str):
        """
        Loads the index from a pickle file.
        """
        logger.info(f"Loading index from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file {path} not found.")
            
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.corpus = data["corpus"]
        logger.info(f"Index loaded with {len(self.corpus)} documents.")
