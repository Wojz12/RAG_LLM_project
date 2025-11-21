import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    RAG Generator using a lightweight, instruction-tuned LLM (TinyLlama/Qwen).
    Optimized for CPU/GPU compatibility without requiring bitsandbytes (4-bit) if incompatible.
    """
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """
        Loads the model. Tries to use GPU with float16 if available, otherwise float32 on CPU.
        Avoids bitsandbytes 4-bit quantization to ensure stability on Python 3.14+.
        """
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

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 150) -> str:
        """
        Generates an answer given the query and retrieved context.
        """
        # Create prompt using the model's chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question using only the provided context. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
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
        
        return answer.strip()
