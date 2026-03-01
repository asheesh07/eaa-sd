"""
Entropy Analyzer — Adaptive K Decision Engine
===============================================
The core intelligence of the system. Computes token-level entropy,
tracks entropy trajectories, and decides when to stop drafting.

Key ideas drawn from:
- EASD (arxiv 2512.23765): entropy-aware rejection with dual-model entropy
- AdaSD (arxiv 2512.11280): adaptive thresholds via entropy + JS divergence
- HeteroSpec: cumulative entropy for context predictability estimation
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


@dataclass
class EntropyState:
    """Tracks entropy statistics across the decoding session."""
    entropy_history: List[float] = field(default_factory=list)
    acceptance_history: List[float] = field(default_factory=list)
    k_history: List[int] = field(default_factory=list)
    
    # EMA trackers
    ema_entropy: float = 1.0
    ema_acceptance_rate: float = 0.8
    ema_js_threshold: float = 0.1
    
    # Current adaptive K
    current_k: int = 5
    
    # Stats
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_verification_rounds: int = 0


class EntropyAnalyzer:
    """
    Computes entropy metrics and makes adaptive K decisions.
    
    The entropy of a probability distribution p is:
        H(p) = -Σ p(x) * log(p(x))
    
    Low entropy = model is confident (peaked distribution)
    High entropy = model is uncertain (flat distribution)
    
    We use this to:
    1. Decide how many tokens to draft (adaptive K)
    2. Decide whether to accept/reject during verification (EASD penalty)
    3. Track session-level statistics for online adaptation
    """
    
    def __init__(self, config):
        self.config = config.entropy
        self.gen_config = config.generation
        self.state = EntropyState(current_k=config.entropy.k_initial)
    
    # ──────────────────────────────────────────────
    #  CORE ENTROPY COMPUTATIONS
    # ──────────────────────────────────────────────
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute Shannon entropy from logits.
        
        Args:
            logits: Raw logits [batch, vocab_size] or [vocab_size]
            temperature: Sampling temperature (applied before softmax)
        
        Returns:
            Entropy scalar (bits)
        """
        # CRITICAL: cast to float32 — fp16/bf16/quantized logits cause NaN in softmax
        logits = logits.detach().float()
        
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # Clamp to prevent overflow before softmax
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        # Apply temperature
        scaled_logits = logits / max(temperature, 1e-8)
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Clamp probs to avoid log(0)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # Shannon entropy: H = -Σ p*log(p)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Safety: replace any remaining NaN/Inf with a default mid-entropy value
        entropy = torch.where(torch.isfinite(entropy), entropy, torch.tensor(2.0))
        
        return entropy.squeeze()
    
    @staticmethod
    def compute_js_divergence(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Jensen-Shannon divergence between two distributions.
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        
        This is symmetric and bounded [0, ln(2)].
        Used for adaptive acceptance thresholds (AdaSD-style).
        """
        # CRITICAL: cast to float32
        p_logits = p_logits.detach().float().clamp(-100, 100)
        q_logits = q_logits.detach().float().clamp(-100, 100)
        
        p = F.softmax(p_logits / max(temperature, 1e-8), dim=-1).clamp(min=1e-10)
        q = F.softmax(q_logits / max(temperature, 1e-8), dim=-1).clamp(min=1e-10)
        
        # Mixture distribution
        m = (0.5 * (p + q)).clamp(min=1e-10)
        
        # KL divergences — use log_target form for numerical stability
        kl_pm = F.kl_div(m.log(), p, reduction='sum', log_target=False)
        kl_qm = F.kl_div(m.log(), q, reduction='sum', log_target=False)
        
        jsd = 0.5 * (kl_pm + kl_qm)
        
        # Safety
        if not torch.isfinite(jsd):
            return torch.tensor(0.0)
        return jsd.clamp(min=0.0)
    
    @staticmethod
    def compute_top_n_overlap(
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        n: int = 5
    ) -> float:
        """
        Fraction of overlap between top-N predictions of two models.
        Used in EASD entropy penalty.
        """
        top_a = torch.topk(logits_a.squeeze(), n).indices
        top_b = torch.topk(logits_b.squeeze(), n).indices
        
        # Set intersection
        overlap = len(set(top_a.tolist()) & set(top_b.tolist()))
        return overlap / n
    
    # ──────────────────────────────────────────────
    #  ADAPTIVE K DECISIONS
    # ──────────────────────────────────────────────
    
    def should_stop_drafting(
        self,
        current_entropy: float,
        draft_position: int,
        entropy_sequence: List[float]
    ) -> bool:
        """
        Decide whether to stop the draft loop early.
        
        Stopping conditions (any triggers halt):
        1. Reached current adaptive K limit
        2. Entropy exceeds high threshold (model is lost)
        3. Entropy spike detected (sudden uncertainty jump)
        4. Cumulative entropy exceeds budget
        
        Args:
            current_entropy: Entropy of the token just generated
            draft_position: How many tokens drafted so far (0-indexed)
            entropy_sequence: All entropies in this draft round
        
        Returns:
            True if drafting should stop
        """
        cfg = self.config
        
        # Condition 1: Reached K limit
        if draft_position + 1 >= self.state.current_k:
            return True
        
        # Condition 2: Absolute entropy threshold
        if current_entropy > cfg.entropy_high:
            return True
        
        # Condition 3: Entropy spike detection
        if len(entropy_sequence) >= 2:
            prev_entropy = entropy_sequence[-2]
            if prev_entropy > 0.01:  # avoid division by near-zero
                spike_ratio = current_entropy / prev_entropy
                if spike_ratio > cfg.entropy_spike_ratio:
                    return True
        
        # Condition 4: Cumulative entropy budget
        # If average entropy of drafted sequence is climbing, stop early
        if len(entropy_sequence) >= 3:
            recent_avg = sum(entropy_sequence[-3:]) / 3
            if recent_avg > (cfg.entropy_low + cfg.entropy_high) / 2:
                return True
        
        return False
    
    def compute_adaptive_k(self) -> int:
        """
        Compute the K for the next draft round based on recent history.
        
        Uses exponential moving average of acceptance rates:
        - High acceptance → increase K (we can draft more aggressively)
        - Low acceptance → decrease K (verify sooner)
        
        Also factors in recent entropy trends.
        """
        cfg = self.config
        
        if self.state.total_verification_rounds == 0:
            return cfg.k_initial
        
        # Get current acceptance EMA
        acc_rate = self.state.ema_acceptance_rate
        
        # Decision logic
        if acc_rate > cfg.acceptance_target + 0.1:
            # Acceptance is great — be more aggressive
            new_k = self.state.current_k + cfg.k_increase_step
        elif acc_rate < cfg.acceptance_target - 0.1:
            # Acceptance is poor — pull back
            new_k = self.state.current_k - cfg.k_decrease_step
        else:
            # In the sweet spot, keep current
            new_k = self.state.current_k
        
        # Also factor in recent entropy
        if self.state.entropy_history:
            recent_entropy = sum(self.state.entropy_history[-5:]) / min(5, len(self.state.entropy_history))
            if recent_entropy < cfg.entropy_low:
                new_k += 1  # Extra bonus for low-entropy contexts
            elif recent_entropy > cfg.entropy_high:
                new_k -= 1  # Extra penalty for high-entropy
        
        # Clamp
        new_k = max(cfg.k_min, min(cfg.k_max, new_k))
        
        return new_k
    
    # ──────────────────────────────────────────────
    #  EASD ENTROPY PENALTY (Verification Phase)
    # ──────────────────────────────────────────────
    
    def easd_should_reject(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> bool:
        """
        EASD-style entropy penalty during verification.
        
        When BOTH models have high entropy AND their top-N predictions 
        substantially overlap, the token is rejected and resampled from 
        the target. This prevents low-confidence errors from propagating.
        
        From arxiv 2512.23765:
        "When both models exhibit high entropy with substantial overlap 
        among their top-N predictions, the corresponding token is rejected 
        and re-sampled by the target LLM."
        """
        if not self.config.easd_enabled:
            return False
        
        cfg = self.config
        
        # Compute entropies for both models
        draft_entropy = self.compute_entropy(draft_logits, temperature).item()
        target_entropy = self.compute_entropy(target_logits, temperature).item()
        
        # Both must have high entropy
        if draft_entropy < cfg.easd_entropy_threshold or target_entropy < cfg.easd_entropy_threshold:
            return False
        
        # Check top-N overlap
        overlap = self.compute_top_n_overlap(
            draft_logits, target_logits, cfg.easd_top_n
        )
        
        # High overlap under high uncertainty = reject
        return overlap >= cfg.easd_overlap_threshold
    
    def js_should_reject(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> bool:
        """
        AdaSD-style JS divergence acceptance criterion.
        
        Accept if JSD(draft || target) < adaptive threshold.
        Reject otherwise.
        
        The threshold is updated via EMA based on recent history.
        """
        if not self.config.js_divergence_enabled:
            return False
        
        jsd = self.compute_js_divergence(
            draft_logits, target_logits, temperature
        ).item()
        
        return jsd > self.state.ema_js_threshold
    
    # ──────────────────────────────────────────────
    #  STATE UPDATES
    # ──────────────────────────────────────────────
    
    def update_after_verification(
        self,
        n_drafted: int,
        n_accepted: int,
        draft_entropies: List[float],
        js_divergences: Optional[List[float]] = None
    ):
        """
        Update internal state after a verification round.
        Called by the speculative engine after each draft-verify cycle.
        """
        cfg = self.config
        
        # Update counters
        self.state.total_draft_tokens += n_drafted
        self.state.total_accepted_tokens += n_accepted
        self.state.total_verification_rounds += 1
        
        # Acceptance rate for this round
        round_acceptance = n_accepted / max(n_drafted, 1)
        self.state.acceptance_history.append(round_acceptance)
        
        # Update EMA acceptance rate
        self.state.ema_acceptance_rate = (
            cfg.ema_alpha * round_acceptance +
            (1 - cfg.ema_alpha) * self.state.ema_acceptance_rate
        )
        
        # Update entropy EMA
        if draft_entropies:
            avg_entropy = sum(draft_entropies) / len(draft_entropies)
            self.state.ema_entropy = (
                cfg.ema_alpha * avg_entropy +
                (1 - cfg.ema_alpha) * self.state.ema_entropy
            )
            self.state.entropy_history.extend(draft_entropies)
        
        # Update JS divergence threshold (AdaSD-style adaptive)
        if js_divergences and cfg.js_divergence_enabled:
            avg_jsd = sum(js_divergences) / len(js_divergences)
            self.state.ema_js_threshold = (
                cfg.js_ema_alpha * avg_jsd +
                (1 - cfg.js_ema_alpha) * self.state.ema_js_threshold
            )
        
        # Compute new adaptive K
        new_k = self.compute_adaptive_k()
        self.state.k_history.append(new_k)
        self.state.current_k = new_k
    
    def get_stats(self) -> dict:
        """Return summary statistics."""
        total = self.state.total_draft_tokens
        accepted = self.state.total_accepted_tokens
        return {
            "total_draft_tokens": total,
            "total_accepted_tokens": accepted,
            "overall_acceptance_rate": accepted / max(total, 1),
            "ema_acceptance_rate": self.state.ema_acceptance_rate,
            "ema_entropy": self.state.ema_entropy,
            "current_k": self.state.current_k,
            "verification_rounds": self.state.total_verification_rounds,
            "avg_k": (
                sum(self.state.k_history) / len(self.state.k_history)
                if self.state.k_history else self.config.k_initial
            ),
        }