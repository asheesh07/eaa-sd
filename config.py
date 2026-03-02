"""
Entropy-Aware Adaptive-K Speculative Decoding — Configuration
==============================================================
All hyperparameters, model paths, and tunable constants.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Model identifiers and loading settings."""
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Device placement
    draft_device: str = "cuda:0"
    target_device: str = "cuda:0"
    
    # Precision
    draft_dtype: torch.dtype = torch.float16
    target_dtype: torch.dtype = torch.float16
    
    # Load with 4-bit quantization for target to fit on single GPU
    use_4bit_target: bool = True
    use_4bit_draft: bool = False


@dataclass
class EntropyConfig:
    """Entropy-aware adaptive K parameters."""
    
    # --- Adaptive K bounds ---
    k_min: int = 2                  # MUST be >= 2 to avoid K=1 trap state
    k_max: int = 12                 # Maximum draft tokens per step
    k_initial: int = 5              # Starting K before any adaptation
    
    # --- Entropy thresholds ---
    # Calibrated for 32K+ vocab LLMs (max entropy ≈ 10.37 nats for 32K vocab)
    # Typical confident token: 0.5-2.0, uncertain: 3-6, confused: 6+
    entropy_low: float = 1.5        # Below this = very confident, keep drafting
    entropy_high: float = 6.0       # Above this = genuinely confused, stop and verify
    entropy_spike_ratio: float = 3.0  # Absolute delta (nats) that triggers spike stop
    
    # --- Adaptive K update (EMA-based) ---
    ema_alpha: float = 0.2          # Slower EMA for stability
    acceptance_target: float = 0.45  # Realistic for misaligned small→large pairs
    k_increase_step: int = 1        # Symmetric: +1
    k_decrease_step: int = 1        # Symmetric: -1
    
    # --- EASD-style entropy penalty ---
    easd_enabled: bool = True
    easd_entropy_threshold: float = 4.0
    easd_top_n: int = 5
    easd_overlap_threshold: float = 0.6
    
    # --- Jensen-Shannon divergence threshold ---
    js_divergence_enabled: bool = True
    js_threshold_initial: float = 0.1
    js_ema_alpha: float = 0.2


@dataclass
class GenerationConfig:
    """Generation parameters."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Sampling vs greedy
    do_sample: bool = True
    
    # Stopping
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


@dataclass
class SpeculativeConfig:
    """Top-level config combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Logging
    verbose: bool = True
    log_entropy: bool = True
    log_acceptance: bool = True
    
    # Benchmarking
    warmup_steps: int = 3
    benchmark_runs: int = 5