"""
Draft Model — TinyLlama 1.1B Wrapper
======================================
Handles loading, KV-cache management, and token generation
with entropy computation at each step.

TinyLlama 1.1B is ~4.4x smaller than Llama-2 7B, making it
an excellent draft model — fast enough to speculate aggressively
while sharing the same tokenizer family (LlamaTokenizer).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, List, NamedTuple
from entropy_analyzer import EntropyAnalyzer


class DraftOutput(NamedTuple):
    """Output of a single draft step."""
    token_id: int
    logits: torch.Tensor        # [vocab_size] raw logits
    entropy: float              # Shannon entropy of this step
    log_prob: float             # Log probability of the sampled token


class DraftModel:
    """
    TinyLlama 1.1B draft model.
    
    Responsibilities:
    - Fast autoregressive generation of candidate tokens
    - Entropy computation at each step for adaptive K
    - KV-cache management for efficient sequential generation
    - Providing full logit distributions for verification
    """
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.model
        self.gen_config = config.generation
        self.device = self.model_config.draft_device
        
        print(f"[Draft] Loading {self.model_config.draft_model_name}...")
        
        # Load tokenizer (shared with target since both are Llama family)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.draft_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        load_kwargs = {
            "torch_dtype": self.model_config.draft_dtype,
            "device_map": self.device,
            "trust_remote_code": True,
        }
        if self.model_config.use_4bit_draft:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.draft_model_name,
            **load_kwargs
        )
        self.model.eval()
        
        # Set generation token IDs
        self.gen_config.eos_token_id = self.tokenizer.eos_token_id
        self.gen_config.pad_token_id = self.tokenizer.pad_token_id
        
        # KV-cache state
        self._past_key_values = None
        self._cache_seq_len = 0
        
        print(f"[Draft] TinyLlama 1.1B loaded on {self.device}")
    
    # ──────────────────────────────────────────────
    #  CACHE MANAGEMENT
    # ──────────────────────────────────────────────
    
    def reset_cache(self):
        """Clear the KV cache."""
        self._past_key_values = None
        self._cache_seq_len = 0
    
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Process the prompt through the model and cache KV states.
        
        Args:
            input_ids: [1, seq_len] tensor of token IDs
            
        Returns:
            logits for the last position [1, vocab_size]
        """
        self.reset_cache()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                use_cache=True
            )
        
        self._past_key_values = outputs.past_key_values
        self._cache_seq_len = input_ids.shape[1]
        
        return outputs.logits[:, -1, :]  # [1, vocab_size]
    
    # ──────────────────────────────────────────────
    #  SINGLE-STEP GENERATION
    # ──────────────────────────────────────────────
    
    @torch.no_grad()
    def generate_step(self, token_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate logits for a single next token, using KV cache.
        
        Args:
            token_id: [1, 1] tensor — the last generated token
            
        Returns:
            (logits [1, vocab_size], updated_past_key_values)
        """
        outputs = self.model(
            input_ids=token_id.to(self.device),
            past_key_values=self._past_key_values,
            use_cache=True
        )
        
        self._past_key_values = outputs.past_key_values
        self._cache_seq_len += 1
        
        return outputs.logits[:, -1, :]  # [1, vocab_size]
    
    def sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None
    ) -> Tuple[int, float]:
        """
        Sample a token from logits with temperature, top-p, top-k.
        
        Returns:
            (token_id, log_probability)
        """
        temp = temperature or self.gen_config.temperature
        tp = top_p or self.gen_config.top_p
        tk = top_k or self.gen_config.top_k
        
        logits = logits.squeeze(0).float()  # [vocab_size] — MUST be float32
        
        # Clamp to prevent overflow
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        if not self.gen_config.do_sample:
            # Greedy
            token_id = logits.argmax().item()
            log_prob = F.log_softmax(logits, dim=-1)[token_id].item()
            return token_id, log_prob
        
        # Temperature scaling
        scaled_logits = logits / max(temp, 1e-8)
        
        # Top-K filtering
        if tk > 0:
            top_k_vals, _ = torch.topk(scaled_logits, min(tk, scaled_logits.size(-1)))
            threshold = top_k_vals[-1]
            scaled_logits[scaled_logits < threshold] = float('-inf')
        
        # Top-P (nucleus) filtering
        if tp < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= tp
            sorted_logits[sorted_mask] = float('-inf')
            
            # Scatter back
            scaled_logits = torch.zeros_like(scaled_logits).scatter(
                0, sorted_indices, sorted_logits
            )
        
        # Sample
        probs = F.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log(probs[token_id] + 1e-10).item()
        
        return token_id, log_prob
    
    # ──────────────────────────────────────────────
    #  DRAFT GENERATION (single token with entropy)
    # ──────────────────────────────────────────────
    
    def draft_one(self, prev_token: torch.Tensor) -> DraftOutput:
        """
        Draft a single token and compute its entropy.
        
        This is the atomic unit of the draft loop.
        
        Args:
            prev_token: [1, 1] tensor of the previous token
            
        Returns:
            DraftOutput with token_id, logits, entropy, log_prob
        """
        # Forward pass
        logits = self.generate_step(prev_token)
        
        # Compute entropy
        entropy = EntropyAnalyzer.compute_entropy(
            logits, self.gen_config.temperature
        ).item()
        
        # Sample
        token_id, log_prob = self.sample_token(logits)
        
        return DraftOutput(
            token_id=token_id,
            logits=logits.squeeze(0),  # [vocab_size]
            entropy=entropy,
            log_prob=log_prob
        )
    
    def rollback_cache(self, n_tokens: int):
        """
        Roll back the KV cache by n_tokens.
        Used when verification rejects some drafted tokens.
        Handles both DynamicCache and legacy tuple format.
        """
        if self._past_key_values is None or n_tokens <= 0:
            return
        
        try:
            from transformers.cache_utils import DynamicCache
            if isinstance(self._past_key_values, DynamicCache):
                new_cache = DynamicCache()
                for layer_idx in range(len(self._past_key_values)):
                    key = self._past_key_values.key_cache[layer_idx]
                    value = self._past_key_values.value_cache[layer_idx]
                    if key is not None and key.shape[2] > n_tokens:
                        new_cache.update(
                            key[:, :, :-n_tokens, :],
                            value[:, :, :-n_tokens, :],
                            layer_idx
                        )
                    elif key is not None:
                        new_cache.update(
                            key[:, :, :0, :],
                            value[:, :, :0, :],
                            layer_idx
                        )
                self._past_key_values = new_cache
                self._cache_seq_len -= n_tokens
                return
        except (ImportError, AttributeError):
            pass
        
        # Legacy tuple format
        if isinstance(self._past_key_values, (tuple, list)):
            new_past = []
            for layer_past in self._past_key_values:
                if layer_past is None:
                    new_past.append(None)
                    continue
                truncated_layer = []
                for t in layer_past:
                    if t is None:
                        truncated_layer.append(None)
                    elif t.dim() >= 3 and t.shape[2] > n_tokens:
                        truncated_layer.append(t[:, :, :-n_tokens, :])
                    else:
                        truncated_layer.append(t)
                new_past.append(tuple(truncated_layer))
            self._past_key_values = tuple(new_past)
        
        self._cache_seq_len -= n_tokens
    
    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size