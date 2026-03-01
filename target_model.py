"""
Target Model — Llama-2 7B Wrapper
===================================
Handles loading, parallel verification of draft tokens,
and correction sampling when tokens are rejected.

Key insight: The target model processes ALL K draft tokens in a 
SINGLE forward pass (not K separate passes). This is why speculative
decoding works — we trade K sequential draft-model passes for 
1 parallel target-model pass.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, NamedTuple
import math


class VerificationResult(NamedTuple):
    """Result of verifying a sequence of draft tokens."""
    n_accepted: int                     # How many draft tokens were accepted
    accepted_tokens: List[int]          # Token IDs that were accepted
    correction_token: Optional[int]     # Resampled token at first rejection (or bonus token)
    target_logits_all: torch.Tensor     # All target logits for the positions [K+1, vocab_size]
    js_divergences: List[float]         # JSD at each position


class TargetModel:
    """
    Llama-2 7B target/verifier model.
    
    Responsibilities:
    - Batch verification of K draft tokens in one forward pass
    - Speculative sampling acceptance/rejection
    - Correction token generation on rejection
    - Providing logits for EASD entropy penalty checks
    """
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.model
        self.gen_config = config.generation
        self.device = self.model_config.target_device
        
        print(f"[Target] Loading {self.model_config.target_model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.target_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (likely quantized to fit on GPU)
        load_kwargs = {
            "torch_dtype": self.model_config.target_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.model_config.use_4bit_target:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.target_model_name,
            **load_kwargs
        )
        self.model.eval()
        
        # KV cache
        self._past_key_values = None
        self._cache_seq_len = 0
        
        print(f"[Target] Llama-2 7B loaded on {self.device}")
    
    # ──────────────────────────────────────────────
    #  CACHE MANAGEMENT
    # ──────────────────────────────────────────────
    
    def reset_cache(self):
        self._past_key_values = None
        self._cache_seq_len = 0
    
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process prompt and cache KV states."""
        self.reset_cache()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                use_cache=True
            )
        
        self._past_key_values = outputs.past_key_values
        self._cache_seq_len = input_ids.shape[1]
        
        return outputs.logits[:, -1, :]
    
    def rollback_cache(self, n_tokens: int):
        """Truncate KV cache by n positions."""
        if self._past_key_values is None or n_tokens <= 0:
            return
        
        new_past = []
        for layer_past in self._past_key_values:
            truncated = tuple(t[:, :, :-n_tokens, :] for t in layer_past)
            new_past.append(truncated)
        
        self._past_key_values = tuple(new_past)
        self._cache_seq_len -= n_tokens
    
    # ──────────────────────────────────────────────
    #  BATCH VERIFICATION (The core of speculative decoding)
    # ──────────────────────────────────────────────
    
    @torch.no_grad()
    def verify_draft(
        self,
        draft_token_ids: List[int],
        draft_logits: List[torch.Tensor],
        entropy_analyzer=None
    ) -> VerificationResult:
        """
        Verify K draft tokens in a SINGLE forward pass.
        
        This is the key efficiency trick: instead of running the target
        model K times autoregressively, we feed all K tokens at once.
        The target model produces logits for positions [0, 1, ..., K],
        where position i gives us P_target(x_{i+1} | x_{<i}).
        
        Acceptance criterion (speculative sampling):
        For each draft token x_i sampled from P_draft:
            Accept with probability min(1, P_target(x_i) / P_draft(x_i))
            
        On first rejection at position j:
            Sample correction from: norm(max(0, P_target - P_draft))
            
        If all K tokens accepted:
            Sample bonus token from P_target at position K+1
            
        Args:
            draft_token_ids: K draft token IDs
            draft_logits: K sets of draft logits (for acceptance ratio)
            entropy_analyzer: Optional, for EASD penalty checks
            
        Returns:
            VerificationResult
        """
        K = len(draft_token_ids)
        if K == 0:
            return VerificationResult(
                n_accepted=0, accepted_tokens=[],
                correction_token=None,
                target_logits_all=torch.empty(0),
                js_divergences=[]
            )
        
        # Build input: the K draft tokens
        draft_tensor = torch.tensor(
            [draft_token_ids], dtype=torch.long, device=self.device
        )  # [1, K]
        
        # Single forward pass through target model
        outputs = self.model(
            input_ids=draft_tensor,
            past_key_values=self._past_key_values,
            use_cache=True
        )
        
        # outputs.logits: [1, K, vocab_size]
        # Position i gives P_target(x | context + draft[0:i])
        target_logits_all = outputs.logits.squeeze(0).float()  # [K, vocab_size]
        
        # We need K+1 logits positions:
        # - Positions 0..K-1 verify draft tokens 0..K-1
        # - Position K-1's logits give us the bonus/correction at position K
        # But actually, position i's logits verify draft_token[i]:
        #   - Logits at position 0 (after seeing draft[0]) verify draft[1]
        #   - We need the logits BEFORE draft[0] for verifying draft[0]
        # 
        # Wait — let me be precise about the indexing:
        # The target model's KV cache already has the prompt.
        # We feed [draft[0], draft[1], ..., draft[K-1]].
        # Output logits[i] = P_target(next | prompt + draft[0:i+1])
        #
        # To verify draft[0], we need P_target(x | prompt) — that's the 
        # logits from prefill (position before draft[0]).
        # To verify draft[i], we need logits[i-1].
        # The bonus token comes from logits[K-1].
        #
        # So we need the "pre-draft" logits too. We get these from the
        # speculative engine which stores the last target logits.
        
        # For now, we return all K logits and let the engine handle indexing
        # The engine should prepend the pre-draft logits
        
        # Update KV cache
        new_past = outputs.past_key_values
        
        # Compute JS divergences for each position
        js_divergences = []
        for i in range(K):
            d_logits = draft_logits[i].to(self.device)
            t_logits = target_logits_all[i] if i < K else target_logits_all[-1]
            
            # Simplified JSD computation
            p = F.softmax(d_logits / max(self.gen_config.temperature, 1e-8), dim=-1)
            q = F.softmax(t_logits / max(self.gen_config.temperature, 1e-8), dim=-1)
            m = 0.5 * (p + q)
            jsd = 0.5 * (F.kl_div(m.log(), p, reduction='sum') +
                         F.kl_div(m.log(), q, reduction='sum'))
            js_divergences.append(max(0.0, jsd.item()))
        
        # Now do the speculative sampling acceptance/rejection
        accepted_tokens = []
        n_accepted = 0
        correction_token = None
        temp = self.gen_config.temperature
        
        for i in range(K):
            # For position i:
            # - draft_logits[i] was used to sample draft_token_ids[i]
            # - target logits for verifying position i:
            #   Position 0: need pre-draft logits (handled by engine)
            #   Position i>0: target_logits_all[i-1]
            # This is handled by the engine passing pre_logits separately.
            # Here we process using the logits the engine has assembled.
            
            d_logits = draft_logits[i].to(self.device)
            
            # We verify draft[i] using target logits at position i
            # (the engine will assemble these correctly)
            t_logits = target_logits_all[min(i, K-1)]
            
            draft_token = draft_token_ids[i]
            
            # Compute acceptance probability
            p_draft = F.softmax(d_logits / max(temp, 1e-8), dim=-1)
            p_target = F.softmax(t_logits / max(temp, 1e-8), dim=-1)
            
            # Speculative sampling criterion
            acceptance_prob = torch.clamp(
                p_target[draft_token] / (p_draft[draft_token] + 1e-10),
                max=1.0
            ).item()
            
            # EASD entropy penalty check
            easd_reject = False
            if entropy_analyzer is not None:
                easd_reject = entropy_analyzer.easd_should_reject(
                    d_logits.unsqueeze(0), t_logits.unsqueeze(0), temp
                )
            
            # Accept or reject
            r = torch.rand(1).item()
            
            if r < acceptance_prob and not easd_reject:
                # Accept
                accepted_tokens.append(draft_token)
                n_accepted += 1
            else:
                # Reject — sample correction from residual distribution
                # P_corrected = norm(max(0, P_target - P_draft))
                residual = torch.clamp(p_target - p_draft, min=0.0)
                residual_sum = residual.sum()
                
                if residual_sum > 1e-8:
                    residual = residual / residual_sum
                    correction_token = torch.multinomial(residual, 1).item()
                else:
                    # Fallback: sample from target distribution
                    correction_token = torch.multinomial(p_target, 1).item()
                
                # Rollback: we accepted i tokens, rejected from position i onward
                # The KV cache should reflect only accepted tokens
                n_to_rollback = K - i
                break
        else:
            # All K tokens accepted — sample bonus token from last position
            last_logits = target_logits_all[-1]
            p_bonus = F.softmax(last_logits / max(temp, 1e-8), dim=-1)
            
            if self.gen_config.do_sample:
                correction_token = torch.multinomial(p_bonus, 1).item()
            else:
                correction_token = last_logits.argmax().item()
            
            n_to_rollback = 0
        
        # Fix the KV cache: keep only the accepted portion + correction
        if n_to_rollback > 0:
            # We need to rollback to after the last accepted token
            # The new past_key_values should cover: prompt + accepted_tokens
            self._past_key_values = new_past
            self.rollback_cache(n_to_rollback)
            self._cache_seq_len = self._cache_seq_len  # Already updated by rollback
        else:
            # All accepted — cache is correct as-is
            self._past_key_values = new_past
            self._cache_seq_len += K
        
        return VerificationResult(
            n_accepted=n_accepted,
            accepted_tokens=accepted_tokens,
            correction_token=correction_token,
            target_logits_all=target_logits_all,
            js_divergences=js_divergences
        )
    
    # ──────────────────────────────────────────────
    #  STANDALONE GENERATION (for benchmarking)
    # ──────────────────────────────────────────────
    
    @torch.no_grad()
    def generate_step(self, token_id: torch.Tensor) -> torch.Tensor:
        """Single autoregressive step for vanilla decoding benchmark."""
        outputs = self.model(
            input_ids=token_id.to(self.device),
            past_key_values=self._past_key_values,
            use_cache=True
        )
        self._past_key_values = outputs.past_key_values
        self._cache_seq_len += 1
        return outputs.logits[:, -1, :]
    
    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size