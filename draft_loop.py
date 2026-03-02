"""
Draft Generation Loop — Adaptive Entropy-Aware Drafting (v3)
=============================================================
Generates K candidate tokens from the draft model, dynamically
stopping based on entropy signals.

v3 fixes:
- should_stop_drafting only handles early stopping, not the K limit
- stopped_by correctly classified
- k_min guard is in entropy_analyzer, not duplicated here
"""

import torch
from typing import List, Tuple, NamedTuple
from draft_model import DraftModel, DraftOutput
from entropy_analyzer import EntropyAnalyzer


class DraftSequence(NamedTuple):
    token_ids: List[int]
    logits: List[torch.Tensor]
    entropies: List[float]
    log_probs: List[float]
    stopped_by: str  # "k_limit", "entropy_high", "entropy_spike", "cumulative", "eos"


class DraftLoop:
    
    def __init__(self, draft_model, entropy_analyzer, config):
        self.draft_model = draft_model
        self.entropy_analyzer = entropy_analyzer
        self.config = config
        self.gen_config = config.generation
        self.verbose = config.verbose
    
    def generate_draft(self, last_token_id: int) -> DraftSequence:
        """
        Generate draft tokens with adaptive stopping.
        
        The loop runs for up to current_k steps. At each step AFTER generating
        the token, we ask the entropy analyzer if we should stop early.
        The analyzer handles all the logic about k_min, entropy thresholds, etc.
        """
        token_ids = []
        logits_list = []
        entropies = []
        log_probs = []
        stopped_by = "k_limit"  # Default: we ran the full K
        
        current_k = self.entropy_analyzer.state.current_k
        
        current_token = torch.tensor([[last_token_id]], dtype=torch.long)
        
        for step in range(current_k):
            # Step 1: Generate one draft token
            draft_output = self.draft_model.draft_one(current_token)
            
            # Step 2: Record
            token_ids.append(draft_output.token_id)
            logits_list.append(draft_output.logits.cpu())
            entropies.append(draft_output.entropy)
            log_probs.append(draft_output.log_prob)
            
            # Step 3: Check EOS
            if draft_output.token_id == self.gen_config.eos_token_id:
                stopped_by = "eos"
                break
            
            # Step 4: Ask entropy analyzer about early stopping
            # This only returns True for EARLY stopping (before reaching K).
            # It respects k_min internally and won't stop before enough tokens.
            should_stop = self.entropy_analyzer.should_stop_drafting(
                current_entropy=draft_output.entropy,
                draft_position=step,
                entropy_sequence=entropies
            )
            
            if should_stop:
                # Classify the stop reason
                cfg = self.entropy_analyzer.config
                if draft_output.entropy > cfg.entropy_high:
                    stopped_by = "entropy_high"
                elif (len(entropies) >= 2 and 
                      (entropies[-1] - entropies[-2]) > cfg.entropy_spike_ratio):
                    stopped_by = "entropy_spike"
                else:
                    stopped_by = "cumulative"
                break
            
            # Step 5: Prepare next
            current_token = torch.tensor([[draft_output.token_id]], dtype=torch.long)
        
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
    
    def generate_draft_greedy(self, last_token_id: int, fixed_k: int) -> DraftSequence:
        """Fixed-K draft without adaptive stopping. For benchmarking."""
        token_ids = []
        logits_list = []
        entropies = []
        log_probs = []
        
        current_token = torch.tensor([[last_token_id]], dtype=torch.long)
        
        for _ in range(fixed_k):
            draft_output = self.draft_model.draft_one(current_token)
            token_ids.append(draft_output.token_id)
            logits_list.append(draft_output.logits.cpu())
            entropies.append(draft_output.entropy)
            log_probs.append(draft_output.log_prob)
            
            if draft_output.token_id == self.gen_config.eos_token_id:
                break
            current_token = torch.tensor([[draft_output.token_id]], dtype=torch.long)
        
        return DraftSequence(
            token_ids=token_ids,
            logits=logits_list,
            entropies=entropies,
            log_probs=log_probs,
            stopped_by="fixed_k"
        )