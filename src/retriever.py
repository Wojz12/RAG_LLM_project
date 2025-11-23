import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import pickle
import os
import re
from tqdm import tqdm
import numpy as np
import torch

# Optional import for BM25 to avoid crashes if not used but kept for backward compatibility
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Imports for Dense Retrieval
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

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
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed.")
        self.bm25 = None
        self.corpus = [] 
        
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def build_index(self, corpus: List[Dict[str, Any]]):
        logger.info(f"Building BM25 index for {len(corpus)} passages...")
        self.corpus = corpus
        tokenized_corpus = []
        for doc in tqdm(corpus, desc="Tokenizing corpus"):
             tokenized_corpus.append(self._tokenize(doc["text"]))
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("Index not built!")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        if len(scores) > top_k:
            top_n_indices = np.argpartition(scores, -top_k)[-top_k:]
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
        logger.info(f"Saving index to {path}...")
        data = {"bm25": self.bm25, "corpus": self.corpus}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Index saved.")

    def load_index(self, path: str):
        logger.info(f"Loading index from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file {path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.corpus = data["corpus"]
        logger.info(f"Index loaded with {len(self.corpus)} documents.")


class DenseRetriever(BaseRetriever):
    """
    Dense Retriever using FAISS and Sentence Transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None or faiss is None:
            raise ImportError("sentence-transformers and faiss-cpu are required for DenseRetriever.")
            
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []
        
        # Use GPU for embedding if available (SentenceTransformer handles this mostly, but we check)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)

    def build_index(self, corpus: List[Dict[str, Any]]):
        """
        Encodes corpus and builds FAISS index.
        """
        logger.info(f"Encoding {len(corpus)} passages with {self.model_name}...")
        self.corpus = corpus
        
        texts = [doc["text"] for doc in corpus]
        
        # Encode in batches
        embeddings = self.encoder.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True # Important for cosine similarity
        )
        
        # Initialize FAISS Index
        # Inner Product (IP) corresponds to Cosine Similarity if normalized
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.index:
            raise ValueError("Index not built!")
            
        # Encode query
        query_embedding = self.encoder.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # Search
        # D = distances (scores), I = indices
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        # I[0] contains indices for the first (and only) query
        for rank, idx in enumerate(I[0]):
            if idx == -1: continue # FAISS padding
            doc = self.corpus[idx].copy()
            doc["score"] = float(D[0][rank])
            results.append(doc)
            
        return results

    def save_index(self, path: str):
        """
        Saves FAISS index and corpus separately.
        Path should be the base name/dir.
        """
        logger.info(f"Saving Dense index to {path}...")
        
        # Save FAISS index
        index_file = path + ".faiss"
        faiss.write_index(self.index, index_file)
        
        # Save corpus and metadata
        meta_file = path + ".meta.pkl"
        with open(meta_file, "wb") as f:
            pickle.dump({"corpus": self.corpus, "model_name": self.model_name}, f)
            
        logger.info("Dense Index saved.")

    def load_index(self, path: str):
        logger.info(f"Loading Dense index from {path}...")
        
        index_file = path + ".faiss"
        meta_file = path + ".meta.pkl"
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
             raise FileNotFoundError(f"Index files not found at {path} (.faiss/.meta.pkl)")
             
        self.index = faiss.read_index(index_file)
        
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
            self.corpus = data["corpus"]
            # We assume model_name is compatible or same
            
        logger.info(f"Dense Index loaded with {len(self.corpus)} documents.")
