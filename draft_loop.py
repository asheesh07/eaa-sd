"""
Draft Generation Loop — Adaptive Entropy-Aware Drafting
=========================================================
The inner loop that generates K candidate tokens from the draft model,
dynamically stopping based on entropy signals.

This is where the "adaptive K" magic happens:
- Start generating tokens from the draft model
- At each step, compute entropy
- If entropy is low → keep going (model is confident)
- If entropy spikes or crosses threshold → STOP and hand off to verifier

The loop produces a "draft sequence" that gets batch-verified
by the target model in a single forward pass.
"""

import torch
from typing import List, Tuple, NamedTuple
from draft_model import DraftModel, DraftOutput
from entropy_analyzer import EntropyAnalyzer


class DraftSequence(NamedTuple):
    """Complete output of a draft generation round."""
    token_ids: List[int]                # K draft token IDs
    logits: List[torch.Tensor]          # K logit vectors [vocab_size]
    entropies: List[float]              # K entropy values
    log_probs: List[float]              # K log probabilities
    stopped_by: str                     # Why we stopped: "k_limit", "entropy_high", 
                                        # "entropy_spike", "cumulative", "eos"


class DraftLoop:
    """
    The adaptive draft generation loop.
    
    Flow for each round:
    ┌─────────────┐
    │ Get current K│ ← from EntropyAnalyzer
    └──────┬──────┘
           ▼
    ┌─────────────────────────────────────────┐
    │ FOR i = 0 to K-1:                       │
    │   1. Draft model generates next token   │
    │   2. Compute entropy H(p_draft)         │
    │   3. Check stopping conditions:         │
    │      - H > threshold? → STOP            │
    │      - H spiked? → STOP                 │
    │      - Cumulative H budget? → STOP      │
    │   4. If not stopped, continue           │
    └──────┬──────────────────────────────────┘
           ▼
    ┌─────────────────────────────────────────┐
    │ Return DraftSequence (tokens + logits)  │
    │ → handed to target model for verify     │
    └─────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        draft_model: DraftModel,
        entropy_analyzer: EntropyAnalyzer,
        config
    ):
        self.draft_model = draft_model
        self.entropy_analyzer = entropy_analyzer
        self.config = config
        self.gen_config = config.generation
        self.verbose = config.verbose
    
    def generate_draft(
        self,
        last_token_id: int,
    ) -> DraftSequence:
        """
        Generate a sequence of draft tokens with adaptive stopping.
        
        This is the main draft loop. It:
        1. Gets the current adaptive K from the entropy analyzer
        2. Generates tokens one-by-one from the draft model
        3. Monitors entropy at each step
        4. Stops early if entropy signals suggest uncertainty
        
        Args:
            last_token_id: The last confirmed token (accepted or correction)
            
        Returns:
            DraftSequence containing all drafted tokens and metadata
        """
        token_ids = []
        logits_list = []
        entropies = []
        log_probs = []
        stopped_by = "k_limit"
        
        # Current adaptive K
        current_k = self.entropy_analyzer.state.current_k
        
        # Track the token to feed into next step
        current_token = torch.tensor(
            [[last_token_id]], dtype=torch.long
        )
        
        for step in range(current_k):
            # ── Step 1: Generate one draft token ──
            draft_output = self.draft_model.draft_one(current_token)
            
            # ── Step 2: Record outputs ──
            token_ids.append(draft_output.token_id)
            logits_list.append(draft_output.logits.cpu())  # Store on CPU to save VRAM
            entropies.append(draft_output.entropy)
            log_probs.append(draft_output.log_prob)
            
            # ── Step 3: Check EOS ──
            if draft_output.token_id == self.gen_config.eos_token_id:
                stopped_by = "eos"
                break
            
            # ── Step 4: Entropy-based stopping ──
            should_stop = self.entropy_analyzer.should_stop_drafting(
                current_entropy=draft_output.entropy,
                draft_position=step,
                entropy_sequence=entropies
            )
            
            if should_stop and step + 1 >= self.entropy_analyzer.config.k_min:
                # Only stop early if we've met the minimum K
                if draft_output.entropy > self.entropy_analyzer.config.entropy_high:
                    stopped_by = "entropy_high"
                elif len(entropies) >= 2 and entropies[-1] / max(entropies[-2], 0.01) > self.entropy_analyzer.config.entropy_spike_ratio:
                    stopped_by = "entropy_spike"
                else:
                    stopped_by = "cumulative"
                break
            
            # ── Step 5: Prepare for next step ──
            current_token = torch.tensor(
                [[draft_output.token_id]], dtype=torch.long
            )
        
        if self.verbose:
            avg_entropy = sum(entropies) / max(len(entropies), 1)
            print(
                f"  [Draft] K={len(token_ids)}/{current_k} | "
                f"avg_H={avg_entropy:.3f} | "
                f"stopped_by={stopped_by} | "
                f"entropies={[f'{e:.2f}' for e in entropies]}"
            )
        
        return DraftSequence(
            token_ids=token_ids,
            logits=logits_list,
            entropies=entropies,
            log_probs=log_probs,
            stopped_by=stopped_by
        )
    
    def generate_draft_greedy(
        self,
        last_token_id: int,
        fixed_k: int
    ) -> DraftSequence:
        """
        Generate a fixed-K draft sequence WITHOUT adaptive stopping.
        Used for benchmarking / ablation against the adaptive version.
        """
        token_ids = []
        logits_list = []
        entropies = []
        log_probs = []
        
        current_token = torch.tensor(
            [[last_token_id]], dtype=torch.long
        )
        
        for _ in range(fixed_k):
            draft_output = self.draft_model.draft_one(current_token)
            
            token_ids.append(draft_output.token_id)
            logits_list.append(draft_output.logits.cpu())
            entropies.append(draft_output.entropy)
            log_probs.append(draft_output.log_prob)
            
            if draft_output.token_id == self.gen_config.eos_token_id:
                break
            
            current_token = torch.tensor(
                [[draft_output.token_id]], dtype=torch.long
            )
        
        return DraftSequence(
            token_ids=token_ids,
            logits=logits_list,
            entropies=entropies,
            log_probs=log_probs,
            stopped_by="fixed_k"
        )