"""
Entropy Analyzer v3 — Adaptive K Decision Engine
==================================================
Fixes the K-collapse death spiral from v1/v2.

Root causes of K=1 trap:
1. should_stop_drafting condition 1 fires at step=0 when K=1 → always stops at 1 token
2. compute_adaptive_k requires acc_rate > target+0.05 to increase, but at K=1
   with a misaligned pair, acceptance EMA hovers 0.3-0.5 → never exceeds threshold
3. Asymmetric step sizes (decrease by 2, increase by 1) in v1
4. Entropy thresholds calibrated for char-level models, not 32K-vocab LLMs
5. stopped_by="cumulative" fires on 1 token because cumulative check has no min length

v3 fixes:
- K floor: if K hits k_min, FORCE increase after N rounds regardless of acceptance
- Warmup: K stays at k_initial for first 5 rounds (enough signal to judge)
- Symmetric ±1 steps with bias toward increasing
- Entropy thresholds calibrated for 32K vocab (max ~10.3 nats)
- should_stop_drafting CANNOT fire before generating k_min tokens
- Recovery mechanism: track consecutive rounds at k_min and force bump
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class EntropyState:
    entropy_history: List[float] = field(default_factory=list)
    acceptance_history: List[float] = field(default_factory=list)
    k_history: List[int] = field(default_factory=list)
    ema_entropy: float = 3.0
    ema_acceptance_rate: float = 0.5   # Start neutral, not optimistic
    ema_js_threshold: float = 0.15
    current_k: int = 5
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_verification_rounds: int = 0
    consecutive_at_k_min: int = 0       # Track how long we've been stuck at floor


class EntropyAnalyzer:
    
    def __init__(self, config):
        self.config = config.entropy
        self.gen_config = config.generation
        self.state = EntropyState(current_k=config.entropy.k_initial)
        self._warmup_rounds = 5
    
    # ──────────────────────────────────────────────
    #  CORE COMPUTATIONS  
    # ──────────────────────────────────────────────
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = logits.detach().float()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        scaled = logits / max(temperature, 1e-8)
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.where(torch.isfinite(entropy), entropy, torch.tensor(3.0))
        return entropy.squeeze()
    
    @staticmethod
    def compute_js_divergence(p_logits, q_logits, temperature=1.0):
        p_logits = p_logits.detach().float().clamp(-100, 100)
        q_logits = q_logits.detach().float().clamp(-100, 100)
        # Align vocab sizes
        pv, qv = p_logits.shape[-1], q_logits.shape[-1]
        if pv < qv:
            p_logits = F.pad(p_logits, (0, qv - pv), value=-1e9)
        elif qv < pv:
            q_logits = F.pad(q_logits, (0, pv - qv), value=-1e9)
        p_log = F.log_softmax(p_logits / max(temperature, 1e-8), dim=-1)
        q_log = F.log_softmax(q_logits / max(temperature, 1e-8), dim=-1)
        p, q = p_log.exp(), q_log.exp()
        m = (0.5 * (p + q)).clamp(min=1e-10)
        m_log = m.log()
        kl_pm = (p * (p_log - m_log)).sum(dim=-1)
        kl_qm = (q * (q_log - m_log)).sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        if not torch.isfinite(jsd).all():
            return torch.tensor(0.0)
        return jsd.clamp(min=0.0).squeeze()
    
    @staticmethod
    def compute_top_n_overlap(logits_a, logits_b, n=5):
        top_a = torch.topk(logits_a.detach().float().squeeze(), n).indices
        top_b = torch.topk(logits_b.detach().float().squeeze(), n).indices
        return len(set(top_a.tolist()) & set(top_b.tolist())) / n
    
    # ──────────────────────────────────────────────
    #  ADAPTIVE K — EARLY STOPPING
    # ──────────────────────────────────────────────
    
    def should_stop_drafting(self, current_entropy, draft_position, entropy_sequence):
        """
        Decide whether to stop the draft loop early.
        
        CRITICAL FIX: This can NEVER stop before k_min tokens are generated.
        When K=1, should_stop_drafting should return False at position 0,
        letting the loop complete naturally via the range(K) bound.
        """
        cfg = self.config
        current_k = self.state.current_k
        
        # ── HARD RULE: Never stop before k_min tokens ──
        # This is the primary fix for the K=1 trap.
        # The loop range already limits to current_k tokens.
        # This function should only trigger EARLY stopping WITHIN that range.
        if draft_position + 1 < cfg.k_min:
            return False
        
        # ── Reached current K limit → let loop terminate naturally ──
        # Don't return True here — the for-loop handles this.
        # This function only decides whether to stop EARLY (before K).
        if draft_position + 1 >= current_k:
            return False  # Not "early" stopping — loop will end anyway
        
        # ── Below here: we've generated at least k_min tokens but < current_k ──
        
        # Only apply entropy-based stopping if we have enough tokens to judge
        if draft_position < 2:
            return False
        
        # Condition: Absolute entropy threshold — model is genuinely lost
        if current_entropy > cfg.entropy_high:
            return True
        
        # Condition: Entropy spike — absolute jump 
        if len(entropy_sequence) >= 2:
            delta = current_entropy - entropy_sequence[-2]
            midpoint = (cfg.entropy_low + cfg.entropy_high) / 2
            if delta > cfg.entropy_spike_ratio and current_entropy > midpoint:
                return True
        
        # Condition: Sustained high entropy over recent window
        if len(entropy_sequence) >= 4:
            recent_avg = sum(entropy_sequence[-4:]) / 4
            budget = cfg.entropy_low + 0.75 * (cfg.entropy_high - cfg.entropy_low)
            if recent_avg > budget:
                return True
        
        return False
    
    # ──────────────────────────────────────────────
    #  ADAPTIVE K — COMPUTE NEXT K
    # ──────────────────────────────────────────────
    
    def compute_adaptive_k(self):
        """
        Compute next K with anti-collapse protections.
        
        Key mechanisms:
        1. Warmup: hold at k_initial for first N rounds
        2. Symmetric ±1 steps  
        3. Recovery: if stuck at k_min for 5+ rounds, force bump to k_min+2
        4. Acceptance threshold asymmetry: easy to increase, harder to decrease
        """
        cfg = self.config
        
        # ── Warmup ──
        if self.state.total_verification_rounds < self._warmup_rounds:
            return cfg.k_initial
        
        acc_rate = self.state.ema_acceptance_rate
        current_k = self.state.current_k
        
        # ── Recovery from K floor ──
        # If we've been stuck at k_min for too long, the only way to know
        # if conditions improved is to TRY a higher K.
        if current_k <= cfg.k_min and self.state.consecutive_at_k_min >= 5:
            self.state.consecutive_at_k_min = 0
            return cfg.k_min + 2  # Jump up to probe
        
        # ── Acceptance-based adjustment ──
        # Bias toward increasing: lower bar to increase (+0.03), higher bar to decrease (-0.20)
        if acc_rate > cfg.acceptance_target + 0.03:
            new_k = current_k + 1
        elif acc_rate < cfg.acceptance_target - 0.20:
            new_k = current_k - 1
        else:
            new_k = current_k
        
        # ── Entropy nudge (gentle, only at extremes) ──
        if len(self.state.entropy_history) >= 5:
            recent_ent = sum(self.state.entropy_history[-5:]) / 5
            if recent_ent < cfg.entropy_low * 0.5:
                new_k += 1  # Very confident → draft more
            elif recent_ent > cfg.entropy_high * 0.9:
                new_k -= 1  # Very confused → draft less
        
        new_k = max(cfg.k_min, min(cfg.k_max, new_k))
        
        # Track consecutive rounds at floor
        if new_k <= cfg.k_min:
            self.state.consecutive_at_k_min += 1
        else:
            self.state.consecutive_at_k_min = 0
        
        return new_k
    
    # ──────────────────────────────────────────────
    #  EASD ENTROPY PENALTY
    # ──────────────────────────────────────────────
    
    def easd_should_reject(self, draft_logits, target_logits, temperature=1.0):
        if not self.config.easd_enabled:
            return False
        cfg = self.config
        d_h = self.compute_entropy(draft_logits, temperature).item()
        t_h = self.compute_entropy(target_logits, temperature).item()
        if d_h < cfg.easd_entropy_threshold or t_h < cfg.easd_entropy_threshold:
            return False
        overlap = self.compute_top_n_overlap(draft_logits, target_logits, cfg.easd_top_n)
        return overlap >= cfg.easd_overlap_threshold
    
    def js_should_reject(self, draft_logits, target_logits, temperature=1.0):
        if not self.config.js_divergence_enabled:
            return False
        jsd = self.compute_js_divergence(draft_logits, target_logits, temperature).item()
        return jsd > self.state.ema_js_threshold
    
    # ──────────────────────────────────────────────
    #  STATE UPDATES
    # ──────────────────────────────────────────────
    
    def update_after_verification(self, n_drafted, n_accepted, draft_entropies, js_divergences=None):
        cfg = self.config
        self.state.total_draft_tokens += n_drafted
        self.state.total_accepted_tokens += n_accepted
        self.state.total_verification_rounds += 1
        
        round_acceptance = n_accepted / max(n_drafted, 1)
        self.state.acceptance_history.append(round_acceptance)
        
        alpha = cfg.ema_alpha
        self.state.ema_acceptance_rate = (
            alpha * round_acceptance + (1 - alpha) * self.state.ema_acceptance_rate
        )
        
        if draft_entropies:
            valid = [e for e in draft_entropies if math.isfinite(e)]
            if valid:
                avg = sum(valid) / len(valid)
                self.state.ema_entropy = alpha * avg + (1 - alpha) * self.state.ema_entropy
                self.state.entropy_history.extend(valid)
        
        if js_divergences and cfg.js_divergence_enabled:
            valid_js = [j for j in js_divergences if math.isfinite(j)]
            if valid_js:
                avg_jsd = sum(valid_js) / len(valid_js)
                self.state.ema_js_threshold = (
                    cfg.js_ema_alpha * avg_jsd + (1 - cfg.js_ema_alpha) * self.state.ema_js_threshold
                )
        
        new_k = self.compute_adaptive_k()
        self.state.k_history.append(new_k)
        self.state.current_k = new_k
    
    def get_stats(self):
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