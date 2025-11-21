import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    RAG Generator using a Quantized LLM (e.g., Llama-3-8B or Qwen).
    """
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the model with 4-bit quantization.
        """
        logger.info(f"Loading LLM: {self.model_name}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            # Attempt to load with 4-bit quantization (Preferred)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            logger.info("LLM loaded successfully with 4-bit quantization.")
            
        except Exception as e:
            logger.error(f"Failed to load 4-bit LLM: {e}")
            logger.warning("Environment does not support 4-bit quantization or GPU is missing (likely Python 3.14+ incompatibility with bitsandbytes).")
            logger.warning("Falling back to 'gpt2' (CPU-friendly) to demonstrate RAG pipeline functionality.")
            
            try:
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # GPT-2 doesn't have a pad token by default
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                logger.info("Fallback LLM (gpt2) loaded successfully.")
            except Exception as e2:
                logger.critical(f"Failed to load fallback model: {e2}")
                raise e2

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 50) -> str:
        """
        Generates an answer given the query and retrieved context.
        """
        # Determine max length safe for the model
        model_max_len = getattr(self.tokenizer, "model_max_length", 1024)
        # HuggingFace sometimes returns huge numbers for models without explicit limits
        if model_max_len > 4096: 
            model_max_len = 4096 # Cap at reasonable default if unknown
        if self.model_name == "gpt2":
            model_max_len = 1024
            
        # Format naive prompt to check length
        prompt = self._format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        
        # If too long, truncate context
        if input_len > (model_max_len - max_new_tokens):
            # Calculate how much space we have for context
            # Length of prompt without context
            base_prompt = self._format_prompt(query, "")
            base_len = self.tokenizer(base_prompt, return_tensors="pt")["input_ids"].shape[1]
            
            allowed_context_len = model_max_len - max_new_tokens - base_len - 10 # buffer
            
            # Tokenize context
            context_tokens = self.tokenizer(context)["input_ids"]
            if len(context_tokens) > allowed_context_len:
                # Truncate
                context = self.tokenizer.decode(context_tokens[:allowed_context_len])
                # Re-create prompt
                prompt = self._format_prompt(query, context)
        
        # Final Tokenization
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic for evaluation
                temperature=None, # Explicitly set to None/Default if do_sample is False
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to extract just the answer part
        answer = self._extract_answer(generated_text, prompt)
        return answer

    def _format_prompt(self, query: str, context: str) -> str:
        """
        Formats the prompt for the LLM.
        """
        # Simple RAG Prompt
        prompt = f"""You are a helpful assistant. Answer the question using only the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """
        Extracts the answer from the full generated text.
        """
        # If the model echoes the prompt, strip it.
        # GPT-2 often echoes.
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        
        # Fallback
        return generated_text.replace(prompt, "").strip()
