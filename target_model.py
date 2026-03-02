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
        """Truncate KV cache by n positions. Handles both DynamicCache and tuple formats."""
        if self._past_key_values is None or n_tokens <= 0:
            return
        
        try:
            # Modern transformers (>=4.36) uses DynamicCache object
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
                        # Would truncate everything — just clear
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
    
    # ──────────────────────────────────────────────
    #  BATCH VERIFICATION (The core of speculative decoding)
    # ──────────────────────────────────────────────
    
    @torch.no_grad()
    def verify_draft(
        self,
        draft_token_ids: List[int],
        draft_logits: List[torch.Tensor],
        entropy_analyzer=None,
        pre_draft_logits: torch.Tensor = None
    ) -> VerificationResult:
        """
        Verify K draft tokens in a SINGLE forward pass.
        
        CRITICAL INDEXING:
        After prefill + first_token, the target cache covers [prompt + first_token].
        We feed [draft[0], ..., draft[K-1]] to the target model.
        
        Output logits[i] = P_target(next | cache + draft[0:i+1])
        
        To verify draft[0]: need P_target(next | cache) = pre_draft_logits
        To verify draft[i]: need logits[i-1]  (i >= 1)
        Bonus token: from logits[K-1]
        
        So the "verification logits" for position i are:
            verify_logits[0] = pre_draft_logits
            verify_logits[i] = target_logits_all[i-1]   for i >= 1
            bonus_logits     = target_logits_all[K-1]
        
        Args:
            draft_token_ids: K draft token IDs
            draft_logits: K sets of draft logits
            entropy_analyzer: Optional, for EASD checks
            pre_draft_logits: Target model logits from BEFORE the draft
                              (i.e., P_target(next | prompt + last_confirmed_token))
                              REQUIRED for correct verification of draft[0].
        """
        K = len(draft_token_ids)
        if K == 0:
            return VerificationResult(
                n_accepted=0, accepted_tokens=[],
                correction_token=None,
                target_logits_all=torch.empty(0),
                js_divergences=[]
            )
        
        # ── Forward pass: feed all K draft tokens at once ──
        draft_tensor = torch.tensor(
            [draft_token_ids], dtype=torch.long, device=self.device
        )
        
        outputs = self.model(
            input_ids=draft_tensor,
            past_key_values=self._past_key_values,
            use_cache=True
        )
        
        # target_logits_all[i] = P_target(next | cache + draft[0:i+1])
        target_logits_all = outputs.logits.squeeze(0).float()  # [K, vocab_size]
        new_past = outputs.past_key_values
        target_vocab = target_logits_all.shape[-1]
        
        # ── Build verification logits array ──
        # verify_logits[i] is used to verify draft_token_ids[i]
        # verify_logits[0] = pre_draft_logits (from before any draft token)
        # verify_logits[i] = target_logits_all[i-1] for i >= 1
        
        if pre_draft_logits is not None:
            pre = pre_draft_logits.to(self.device).float().squeeze()
            # Ensure pre_draft_logits matches target vocab size
            if pre.shape[-1] < target_vocab:
                pre = F.pad(pre, (0, target_vocab - pre.shape[-1]), value=-1e9)
            elif pre.shape[-1] > target_vocab:
                pre = pre[..., :target_vocab]
            verify_logits = [pre] + [target_logits_all[i] for i in range(K - 1)]
        else:
            # Fallback: use target_logits_all[i] for verify (off-by-one but no choice)
            verify_logits = [target_logits_all[i] for i in range(K)]
        
        # Bonus logits: always from last position
        bonus_logits = target_logits_all[K - 1]
        
        # ── Helper: align vocab sizes ──
        def align_to_target(draft_l):
            """Pad or truncate draft logits to match target vocab size."""
            draft_l = draft_l.to(self.device).float()
            dv = draft_l.shape[-1]
            if dv < target_vocab:
                return F.pad(draft_l, (0, target_vocab - dv), value=-1e9)
            elif dv > target_vocab:
                return draft_l[..., :target_vocab]
            return draft_l
        
        # ── Compute JS divergences ──
        js_divergences = []
        temp = self.gen_config.temperature
        for i in range(K):
            d_l = align_to_target(draft_logits[i]).clamp(-100, 100)
            t_l = verify_logits[i].clamp(-100, 100)
            p = F.softmax(d_l / max(temp, 1e-8), dim=-1).clamp(min=1e-10)
            q = F.softmax(t_l / max(temp, 1e-8), dim=-1).clamp(min=1e-10)
            m = (0.5 * (p + q)).clamp(min=1e-10)
            jsd = 0.5 * (F.kl_div(m.log(), p, reduction='sum', log_target=False) +
                         F.kl_div(m.log(), q, reduction='sum', log_target=False))
            js_divergences.append(max(0.0, jsd.item()) if torch.isfinite(jsd) else 0.0)
        
        # ── Speculative sampling: accept/reject ──
        accepted_tokens = []
        n_accepted = 0
        correction_token = None
        n_to_rollback = 0
        
        for i in range(K):
            d_l = align_to_target(draft_logits[i])
            t_l = verify_logits[i]
            draft_token = draft_token_ids[i]
            
            p_draft = F.softmax(d_l / max(temp, 1e-8), dim=-1).clamp(min=1e-10)
            p_target = F.softmax(t_l / max(temp, 1e-8), dim=-1).clamp(min=1e-10)
            
            # Acceptance probability: min(1, P_target(x) / P_draft(x))
            # draft_token is always within draft vocab, so index is valid for both
            # (target has >= draft vocab size after padding)
            acceptance_prob = torch.clamp(
                p_target[draft_token] / (p_draft[draft_token] + 1e-10),
                max=1.0
            ).item()
            
            # EASD check
            easd_reject = False
            if entropy_analyzer is not None:
                easd_reject = entropy_analyzer.easd_should_reject(
                    d_l.unsqueeze(0), t_l.unsqueeze(0), temp
                )
            
            r = torch.rand(1).item()
            
            if r < acceptance_prob and not easd_reject:
                accepted_tokens.append(draft_token)
                n_accepted += 1
            else:
                # Correction from residual: norm(max(0, P_target - P_draft))
                residual = torch.clamp(p_target - p_draft, min=0.0)
                residual_sum = residual.sum()
                if residual_sum > 1e-8:
                    correction_token = torch.multinomial(residual / residual_sum, 1).item()
                else:
                    correction_token = torch.multinomial(p_target, 1).item()
                
                n_to_rollback = K - i
                break
        else:
            # All K accepted — bonus token from last target logits
            p_bonus = F.softmax(bonus_logits / max(temp, 1e-8), dim=-1)
            if self.gen_config.do_sample:
                correction_token = torch.multinomial(p_bonus, 1).item()
            else:
                correction_token = bonus_logits.argmax().item()
            n_to_rollback = 0
        
        # ── Fix KV cache ──
        if n_to_rollback > 0:
            self._past_key_values = new_past
            self.rollback_cache(n_to_rollback)
        else:
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