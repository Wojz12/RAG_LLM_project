"""RAG Generator module with hard-grounded LLM generation."""

import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

# Unified grounding constant - use this exact phrase for "I don't know" responses
GROUNDING_FALLBACK = "I don't know from the given documents."


class RAGGenerator:
    """
    RAG Generator using a lightweight, instruction-tuned LLM (TinyLlama/Qwen).
    Optimized for CPU/GPU compatibility without requiring bitsandbytes (4-bit) if incompatible.
    """
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        """Load the model with GPU float16 if available, otherwise CPU float32."""
        logger.info(f"Loading LLM: {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # TinyLlama needs a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            # For 1.1B model, standard fp32 takes ~4GB RAM, fp16 takes ~2GB.
            # This fits easily in most environments.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device
            )
            logger.info("LLM loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 20) -> str:
        """Generate an answer given the query and retrieved context.
        
        Args:
            query: The user's question.
            context: Retrieved and reranked context passages.
            max_new_tokens: Maximum tokens to generate (default 20 for concise answers).
            
        Returns:
            Generated answer string.
        """
        # Truncate context early to save tokens
        context = context[:1500] if len(context) > 1500 else context
        
        # Simple, direct prompt without chat template for better control
        prompt = f"""Extract the answer from the context. Reply with ONLY the answer (1-5 words max).

Context: {context}

Q: Who wrote Romeo and Juliet?
A: William Shakespeare

Q: What is the capital of Japan?
A: Tokyo

Q: {query}
A:"""
        
        # Tokenize directly without chat template for more control
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1900
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated part
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Aggressive cleaning
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "A:", "Answer:", "The answer is:", "The answer is", 
            "It is", "It's", "The ", "I think it's", "I believe it's"
        ]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Take only first line, stop at newline or next Q:
        answer = answer.split('\n')[0].split('Q:')[0].strip()
        
        # Stop at first period or comma for cleaner answers
        if '.' in answer:
            answer = answer.split('.')[0].strip()
        if ',' in answer and len(answer.split(',')[0]) > 3:
            answer = answer.split(',')[0].strip()
        
        return answer
