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

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 30) -> str:
        """Generate an answer given the query and retrieved context.
        
        Args:
            query: The user's question.
            context: Retrieved and reranked context passages.
            max_new_tokens: Maximum tokens to generate (default 30 for concise answers).
            
        Returns:
            Generated answer string.
        """
        # Hard-grounding system prompt optimized for SHORT, DIRECT answers
        system_prompt = (
            "You are a trivia answer bot. Answer questions in 1-5 words ONLY. "
            "Use ONLY the provided context. No explanations, no full sentences. "
            f"If unknown, say: \"{GROUNDING_FALLBACK}\""
        )
        
        # User prompt with examples for few-shot learning
        user_content = f"""Context:
{context}

Examples:
Q: Who wrote Hamlet? A: William Shakespeare
Q: What is the capital of France? A: Paris
Q: When did WW2 end? A: 1945

Question: {query}
A:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template (handles special tokens for TinyLlama/Zephyr/etc.)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and Truncate logic
        # Max context window for TinyLlama is 2048
        max_model_len = 2048
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_len = inputs["input_ids"].shape[1]
        
        if input_len > (max_model_len - max_new_tokens):
            # If prompt is too long, we need to truncate the context part.
            # This is complex with chat templates. simpler heuristic:
            # Re-construct prompt with truncated context string.
            
            # Estimate safe length: Max - System/User overhead (approx 100) - Question len - Output buffer
            q_len = len(self.tokenizer(query)["input_ids"])
            allowed_ctx_len = max_model_len - max_new_tokens - q_len - 200
            
            if allowed_ctx_len < 50: allowed_ctx_len = 50
            
            # Truncate context
            ctx_tokens = self.tokenizer(context)["input_ids"]
            if len(ctx_tokens) > allowed_ctx_len:
                ctx_tokens = ctx_tokens[:allowed_ctx_len]
                context = self.tokenizer.decode(ctx_tokens)
                
                # Re-build prompt
                messages[1]["content"] = f"Context:\n{context}\n\nQuestion: {query}"
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Final Tokenization
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer: The model output includes the prompt + response.
        # We need to parse it. TinyLlama usually outputs just the response if using apply_chat_template correctly?
        # No, generate() returns full sequence.
        
        # Find the start of the assistant response
        # Using simple string splitting based on the last prompt part might be fragile.
        # A robust way is to slice the output tokens.
        
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove common prefixes the model might add
        prefixes_to_remove = ["A:", "Answer:", "The answer is", "The answer is:"]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Take only first line/sentence for conciseness
        answer = answer.split('\n')[0].strip()
        
        # Remove trailing punctuation for cleaner matching
        if answer.endswith('.'):
            answer = answer[:-1].strip()
        
        return answer
