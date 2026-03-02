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
    draft_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    target_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    
    # Device placement
    draft_device: str = "cuda:0"
    target_device: str = "cuda:0"  # same GPU if VRAM allows, else "cuda:1"
    
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
    k_min: int = 1                  # Minimum draft tokens per step
    k_max: int = 12                 # Maximum draft tokens per step
    k_initial: int = 5              # Starting K before any adaptation
    
    # --- Entropy thresholds ---
    # These control when to stop drafting early
    entropy_low: float = 0.5        # Below this = very confident, keep drafting
    entropy_high: float = 2.5       # Above this = uncertain, stop and verify
    entropy_spike_ratio: float = 2.0  # If entropy jumps >2x from previous token, stop
    
    # --- Adaptive K update (EMA-based) ---
    # After each verification round, we update K based on acceptance rate
    ema_alpha: float = 0.3          # Exponential moving average smoothing
    acceptance_target: float = 0.8  # Target acceptance rate
    k_increase_step: int = 1        # How much to increase K when acceptance is high
    k_decrease_step: int = 2        # How much to decrease K when acceptance is low
    
    # --- EASD-style entropy penalty (from arxiv 2512.23765) ---
    # When BOTH models have high entropy AND high top-N overlap, reject and resample
    easd_enabled: bool = True
    easd_entropy_threshold: float = 2.0  # Both models must exceed this
    easd_top_n: int = 5                   # Top-N tokens to check overlap
    easd_overlap_threshold: float = 0.6   # Fraction of top-N that must overlap
    
    # --- Jensen-Shannon divergence threshold (from AdaSD arxiv 2512.11280) ---
    js_divergence_enabled: bool = True
    js_threshold_initial: float = 0.1     # Initial JS divergence threshold
    js_ema_alpha: float = 0.2             # EMA for adaptive JS threshold


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
    eos_token_id: Optional[int] = None  # Set from tokenizer
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