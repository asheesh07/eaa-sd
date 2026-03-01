"""
Speculative Engine — Full Decode Orchestrator
================================================
This is the top-level module that ties everything together:
Draft Loop → Entropy Analyzer → Target Verification → Token Output

The speculative decoding algorithm:
    repeat until max_tokens or EOS:
        1. Draft model generates K tokens (adaptive K via entropy)
        2. Target model verifies all K tokens in ONE forward pass
        3. Accept prefix of tokens that pass speculative sampling
        4. On rejection: resample correction from target's residual
        5. Update entropy analyzer state for next round's K
        6. Append accepted tokens + correction to output

Guarantees: Output distribution is IDENTICAL to sampling from the 
target model alone (when EASD penalty is disabled). This is the 
mathematical beauty of speculative sampling.
"""

import torch
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import SpeculativeConfig
from draft_model import DraftModel
from target_model import TargetModel
from draft_loop import DraftLoop, DraftSequence
from entropy_analyzer import EntropyAnalyzer


@dataclass
class GenerationOutput:
    """Final output of the speculative decoding engine."""
    text: str
    token_ids: List[int]
    total_tokens: int
    wall_time_seconds: float
    tokens_per_second: float
    
    # Speculative stats
    total_draft_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    avg_adaptive_k: float
    verification_rounds: int
    
    # Entropy stats
    entropy_history: List[float]
    k_history: List[int]


class SpeculativeEngine:
    """
    The main engine. Orchestrates the full speculative decode pipeline.
    
    Architecture:
    
    ┌─────────────────────────────────────────────────────────┐
    │                    DECODE LOOP                           │
    │                                                         │
    │   ┌───────────┐   ┌───────────────┐   ┌─────────────┐  │
    │   │  PREFILL   │──▶│  DRAFT LOOP   │──▶│  VERIFY     │  │
    │   │ (both     │   │  (K tokens    │   │  (1 target  │  │
    │   │  models)  │   │   adaptive)   │   │   fwd pass) │  │
    │   └───────────┘   └───────┬───────┘   └──────┬──────┘  │
    │                           │                   │         │
    │                           ▼                   ▼         │
    │                    ┌─────────────┐    ┌──────────────┐  │
    │                    │  ENTROPY    │    │  ACCEPT/     │  │
    │                    │  ANALYZER   │◀───│  REJECT      │  │
    │                    │  (update K) │    │  + CORRECT   │  │
    │                    └─────────────┘    └──────────────┘  │
    │                           │                             │
    │                           ▼                             │
    │                    ┌─────────────┐                      │
    │                    │  OUTPUT     │                      │
    │                    │  TOKENS     │                      │
    │                    └─────────────┘                      │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: SpeculativeConfig = None):
        self.config = config or SpeculativeConfig()
        
        # Initialize components
        print("=" * 60)
        print("  Entropy-Aware Adaptive-K Speculative Decoding")
        print("=" * 60)
        
        self.draft_model = DraftModel(self.config)
        self.target_model = TargetModel(self.config)
        self.entropy_analyzer = EntropyAnalyzer(self.config)
        self.draft_loop = DraftLoop(
            self.draft_model, self.entropy_analyzer, self.config
        )
        
        # Shared tokenizer (both models use Llama tokenizer)
        self.tokenizer = self.draft_model.tokenizer
        
        print("=" * 60)
        print(f"  Draft: {self.config.model.draft_model_name}")
        print(f"  Target: {self.config.model.target_model_name}")
        print(f"  Adaptive K: [{self.config.entropy.k_min}, {self.config.entropy.k_max}]")
        print(f"  EASD penalty: {'ON' if self.config.entropy.easd_enabled else 'OFF'}")
        print(f"  JS adaptive: {'ON' if self.config.entropy.js_divergence_enabled else 'OFF'}")
        print("=" * 60)
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = None) -> GenerationOutput:
        """
        Generate text using entropy-aware adaptive-K speculative decoding.
        
        Args:
            prompt: Input text
            max_new_tokens: Override config max tokens
            
        Returns:
            GenerationOutput with text, stats, and diagnostics
        """
        max_tokens = max_new_tokens or self.config.generation.max_new_tokens
        
        # ── TOKENIZE ──
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]
        
        # ── PREFILL BOTH MODELS ──
        print(f"\n[Engine] Prefilling {prompt_len} prompt tokens...")
        
        self.draft_model.prefill(input_ids)
        target_prefill_logits = self.target_model.prefill(input_ids)
        
        # Sample first token from target model (this seeds the generation)
        if self.config.generation.do_sample:
            first_probs = torch.softmax(
                target_prefill_logits.squeeze() / self.config.generation.temperature,
                dim=-1
            )
            last_token_id = torch.multinomial(first_probs, 1).item()
        else:
            last_token_id = target_prefill_logits.squeeze().argmax().item()
        
        # Also feed this first token to the draft model's cache
        first_token_tensor = torch.tensor(
            [[last_token_id]], dtype=torch.long
        )
        _ = self.draft_model.generate_step(first_token_tensor)
        _ = self.target_model.generate_step(first_token_tensor)
        
        # ── GENERATION STATE ──
        generated_tokens = [last_token_id]
        start_time = time.perf_counter()
        round_num = 0
        
        print(f"[Engine] Starting speculative decode (max {max_tokens} tokens)...\n")
        
        # ── MAIN DECODE LOOP ──
        while len(generated_tokens) < max_tokens:
            round_num += 1
            
            if self.config.verbose:
                print(f"── Round {round_num} | "
                      f"tokens={len(generated_tokens)} | "
                      f"K={self.entropy_analyzer.state.current_k} ──")
            
            # ═══════════════════════════════════════════
            #  PHASE 1: DRAFT (generate K candidate tokens)
            # ═══════════════════════════════════════════
            
            # Roll back draft model cache to after last confirmed token
            # (It may have advanced during the previous draft round)
            # Actually, after verification we handle this — see below
            
            draft_seq = self.draft_loop.generate_draft(
                last_token_id=last_token_id
            )
            
            if len(draft_seq.token_ids) == 0:
                # Shouldn't happen, but safety check
                break
            
            # ═══════════════════════════════════════════
            #  PHASE 2: VERIFY (target model checks all K at once)
            # ═══════════════════════════════════════════
            
            verification = self.target_model.verify_draft(
                draft_token_ids=draft_seq.token_ids,
                draft_logits=draft_seq.logits,
                entropy_analyzer=self.entropy_analyzer
            )
            
            # ═══════════════════════════════════════════
            #  PHASE 3: ACCEPT / REJECT / CORRECT
            # ═══════════════════════════════════════════
            
            n_accepted = verification.n_accepted
            n_drafted = len(draft_seq.token_ids)
            
            # Add accepted tokens to output
            generated_tokens.extend(verification.accepted_tokens)
            
            # Add correction/bonus token
            if verification.correction_token is not None:
                generated_tokens.append(verification.correction_token)
                last_token_id = verification.correction_token
            elif verification.accepted_tokens:
                last_token_id = verification.accepted_tokens[-1]
            
            # ═══════════════════════════════════════════
            #  PHASE 4: SYNC CACHES
            # ═══════════════════════════════════════════
            
            # Draft model needs to be rolled back to match accepted state
            n_rejected = n_drafted - n_accepted
            if n_rejected > 0:
                self.draft_model.rollback_cache(n_rejected)
            
            # Feed the correction token into draft model cache
            if verification.correction_token is not None:
                correction_tensor = torch.tensor(
                    [[verification.correction_token]], dtype=torch.long
                )
                _ = self.draft_model.generate_step(correction_tensor)
                _ = self.target_model.generate_step(correction_tensor)
            
            # ═══════════════════════════════════════════
            #  PHASE 5: UPDATE ENTROPY ANALYZER
            # ═══════════════════════════════════════════
            
            self.entropy_analyzer.update_after_verification(
                n_drafted=n_drafted,
                n_accepted=n_accepted,
                draft_entropies=draft_seq.entropies,
                js_divergences=verification.js_divergences
            )
            
            if self.config.verbose:
                print(
                    f"  [Verify] accepted={n_accepted}/{n_drafted} | "
                    f"correction={'yes' if n_rejected > 0 else 'bonus'} | "
                    f"next_K={self.entropy_analyzer.state.current_k} | "
                    f"ema_acc={self.entropy_analyzer.state.ema_acceptance_rate:.3f}"
                )
                print()
            
            # ── Check EOS ──
            if last_token_id == self.config.generation.eos_token_id:
                break
            
            # ── Check generated enough ──
            if len(generated_tokens) >= max_tokens:
                break
        
        # ── FINALIZE ──
        elapsed = time.perf_counter() - start_time
        total_new = len(generated_tokens)
        
        # Decode
        all_ids = input_ids.squeeze().tolist() + generated_tokens
        full_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        stats = self.entropy_analyzer.get_stats()
        
        output = GenerationOutput(
            text=generated_text,
            token_ids=generated_tokens,
            total_tokens=total_new,
            wall_time_seconds=elapsed,
            tokens_per_second=total_new / max(elapsed, 1e-6),
            total_draft_tokens=stats["total_draft_tokens"],
            total_accepted_tokens=stats["total_accepted_tokens"],
            acceptance_rate=stats["overall_acceptance_rate"],
            avg_adaptive_k=stats["avg_k"],
            verification_rounds=stats["verification_rounds"],
            entropy_history=self.entropy_analyzer.state.entropy_history,
            k_history=self.entropy_analyzer.state.k_history,
        )
        
        self._print_summary(output)
        
        return output
    
    @torch.no_grad()
    def generate_vanilla(self, prompt: str, max_new_tokens: int = None) -> GenerationOutput:
        """
        Vanilla autoregressive decoding using ONLY the target model.
        Used as baseline for benchmarking speedup.
        """
        max_tokens = max_new_tokens or self.config.generation.max_new_tokens
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        self.target_model.prefill(input_ids)
        
        # Sample first token
        first_logits = self.target_model.prefill(input_ids)
        if self.config.generation.do_sample:
            probs = torch.softmax(
                first_logits.squeeze() / self.config.generation.temperature,
                dim=-1
            )
            token_id = torch.multinomial(probs, 1).item()
        else:
            token_id = first_logits.squeeze().argmax().item()
        
        generated_tokens = [token_id]
        start_time = time.perf_counter()
        
        for _ in range(max_tokens - 1):
            token_tensor = torch.tensor([[token_id]], dtype=torch.long)
            logits = self.target_model.generate_step(token_tensor)
            
            if self.config.generation.do_sample:
                probs = torch.softmax(
                    logits.squeeze() / self.config.generation.temperature,
                    dim=-1
                )
                token_id = torch.multinomial(probs, 1).item()
            else:
                token_id = logits.squeeze().argmax().item()
            
            generated_tokens.append(token_id)
            
            if token_id == self.config.generation.eos_token_id:
                break
        
        elapsed = time.perf_counter() - start_time
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return GenerationOutput(
            text=text,
            token_ids=generated_tokens,
            total_tokens=len(generated_tokens),
            wall_time_seconds=elapsed,
            tokens_per_second=len(generated_tokens) / max(elapsed, 1e-6),
            total_draft_tokens=0,
            total_accepted_tokens=0,
            acceptance_rate=0.0,
            avg_adaptive_k=0.0,
            verification_rounds=0,
            entropy_history=[],
            k_history=[],
        )
    
    def _print_summary(self, output: GenerationOutput):
        """Print a nice summary of the generation."""
        print("\n" + "=" * 60)
        print("  GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Tokens generated:    {output.total_tokens}")
        print(f"  Wall time:           {output.wall_time_seconds:.2f}s")
        print(f"  Throughput:          {output.tokens_per_second:.1f} tok/s")
        print(f"  ─────────────────────────────────────")
        print(f"  Draft tokens:        {output.total_draft_tokens}")
        print(f"  Accepted tokens:     {output.total_accepted_tokens}")
        print(f"  Acceptance rate:     {output.acceptance_rate:.1%}")
        print(f"  Avg adaptive K:      {output.avg_adaptive_k:.1f}")
        print(f"  Verification rounds: {output.verification_rounds}")
        print(f"  ─────────────────────────────────────")
        if output.k_history:
            print(f"  K range used:        [{min(output.k_history)}, {max(output.k_history)}]")
        print("=" * 60)