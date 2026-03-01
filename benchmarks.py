"""
Benchmark — Compare Adaptive-K Speculative vs Vanilla vs Fixed-K
===================================================================
Runs multiple prompts across different configurations and reports
throughput, acceptance rates, and speedup factors.
"""

import torch
import time
import json
from typing import List, Dict
from config import SpeculativeConfig, ModelConfig, EntropyConfig, GenerationConfig


# ── BENCHMARK PROMPTS (varied difficulty / entropy profiles) ──
PROMPTS = [
    # Low entropy: factual, predictable
    "The capital of France is",
    "Water boils at a temperature of",
    
    # Medium entropy: explanatory
    "Explain how neural networks learn through backpropagation.",
    "What are the key differences between Python and Rust?",
    
    # High entropy: creative, open-ended
    "Write a short story about a robot that discovers it can dream.",
    "Imagine a world where gravity works in reverse. Describe a typical morning.",
    
    # Code (structured, semi-predictable)
    "Write a Python function that implements binary search on a sorted array.",
    
    # Reasoning (high entropy at decision points)
    "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step.",
]


def run_benchmark(max_tokens: int = 128, save_path: str = "benchmark_results.json"):
    """
    Run the full benchmark suite.
    
    Configurations tested:
    1. Vanilla autoregressive (target model only)
    2. Fixed K=3 speculative decoding
    3. Fixed K=5 speculative decoding
    4. Fixed K=8 speculative decoding
    5. Adaptive K (our method) without EASD
    6. Adaptive K (our method) with EASD + JS (full system)
    """
    
    from speculative_engine import SpeculativeEngine
    from draft_loop import DraftLoop
    from entropy_analyzer import EntropyAnalyzer
    
    results = []
    
    # ── Config: Full adaptive (our method) ──
    config_adaptive = SpeculativeConfig(
        model=ModelConfig(use_4bit_target=True),
        entropy=EntropyConfig(
            k_min=1, k_max=12, k_initial=5,
            easd_enabled=True, js_divergence_enabled=True
        ),
        generation=GenerationConfig(
            max_new_tokens=max_tokens, temperature=0.7, do_sample=True
        ),
        verbose=False,
    )
    
    print("Loading models (this happens once)...")
    engine = SpeculativeEngine(config_adaptive)
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        print(f"{'='*60}")
        
        prompt_results = {"prompt": prompt, "configs": {}}
        
        # ── 1. Vanilla autoregressive ──
        print("  [1/4] Vanilla autoregressive...")
        engine.target_model.reset_cache()
        vanilla = engine.generate_vanilla(prompt, max_tokens)
        prompt_results["configs"]["vanilla"] = {
            "throughput": vanilla.tokens_per_second,
            "wall_time": vanilla.wall_time_seconds,
            "tokens": vanilla.total_tokens,
        }
        
        # ── 2. Adaptive K (full system) ──
        print("  [2/4] Adaptive K + EASD + JS...")
        engine.draft_model.reset_cache()
        engine.target_model.reset_cache()
        engine.entropy_analyzer = EntropyAnalyzer(config_adaptive)
        engine.draft_loop = DraftLoop(
            engine.draft_model, engine.entropy_analyzer, config_adaptive
        )
        
        adaptive = engine.generate(prompt, max_tokens)
        prompt_results["configs"]["adaptive_full"] = {
            "throughput": adaptive.tokens_per_second,
            "wall_time": adaptive.wall_time_seconds,
            "tokens": adaptive.total_tokens,
            "acceptance_rate": adaptive.acceptance_rate,
            "avg_k": adaptive.avg_adaptive_k,
            "verification_rounds": adaptive.verification_rounds,
            "speedup": vanilla.wall_time_seconds / max(adaptive.wall_time_seconds, 1e-6),
        }
        
        # ── 3. Adaptive K without EASD (ablation) ──
        print("  [3/4] Adaptive K (no EASD)...")
        config_no_easd = SpeculativeConfig(
            model=config_adaptive.model,
            entropy=EntropyConfig(
                k_min=1, k_max=12, k_initial=5,
                easd_enabled=False, js_divergence_enabled=False
            ),
            generation=config_adaptive.generation,
            verbose=False,
        )
        engine.draft_model.reset_cache()
        engine.target_model.reset_cache()
        engine.entropy_analyzer = EntropyAnalyzer(config_no_easd)
        engine.draft_loop = DraftLoop(
            engine.draft_model, engine.entropy_analyzer, config_no_easd
        )
        engine.config = config_no_easd
        
        adaptive_no_easd = engine.generate(prompt, max_tokens)
        prompt_results["configs"]["adaptive_no_easd"] = {
            "throughput": adaptive_no_easd.tokens_per_second,
            "wall_time": adaptive_no_easd.wall_time_seconds,
            "tokens": adaptive_no_easd.total_tokens,
            "acceptance_rate": adaptive_no_easd.acceptance_rate,
            "avg_k": adaptive_no_easd.avg_adaptive_k,
            "speedup": vanilla.wall_time_seconds / max(adaptive_no_easd.wall_time_seconds, 1e-6),
        }
        
        # ── 4. Fixed K=5 (standard speculative decoding) ──
        print("  [4/4] Fixed K=5 (standard SD)...")
        config_fixed = SpeculativeConfig(
            model=config_adaptive.model,
            entropy=EntropyConfig(
                k_min=5, k_max=5, k_initial=5,
                easd_enabled=False, js_divergence_enabled=False,
                entropy_high=999.0,  # Effectively disable entropy stopping
            ),
            generation=config_adaptive.generation,
            verbose=False,
        )
        engine.draft_model.reset_cache()
        engine.target_model.reset_cache()
        engine.entropy_analyzer = EntropyAnalyzer(config_fixed)
        engine.draft_loop = DraftLoop(
            engine.draft_model, engine.entropy_analyzer, config_fixed
        )
        engine.config = config_fixed
        
        fixed_k = engine.generate(prompt, max_tokens)
        prompt_results["configs"]["fixed_k5"] = {
            "throughput": fixed_k.tokens_per_second,
            "wall_time": fixed_k.wall_time_seconds,
            "tokens": fixed_k.total_tokens,
            "acceptance_rate": fixed_k.acceptance_rate,
            "speedup": vanilla.wall_time_seconds / max(fixed_k.wall_time_seconds, 1e-6),
        }
        
        results.append(prompt_results)
        
        # Print comparison
        print(f"\n  Results for prompt {i+1}:")
        print(f"  {'Config':<25} {'tok/s':>8} {'accept%':>8} {'speedup':>8}")
        print(f"  {'─'*51}")
        print(f"  {'Vanilla':<25} {vanilla.tokens_per_second:>8.1f} {'N/A':>8} {'1.00x':>8}")
        print(f"  {'Fixed K=5':<25} {fixed_k.tokens_per_second:>8.1f} {fixed_k.acceptance_rate:>7.1%} {prompt_results['configs']['fixed_k5']['speedup']:>7.2f}x")
        print(f"  {'Adaptive (no EASD)':<25} {adaptive_no_easd.tokens_per_second:>8.1f} {adaptive_no_easd.acceptance_rate:>7.1%} {prompt_results['configs']['adaptive_no_easd']['speedup']:>7.2f}x")
        print(f"  {'Adaptive + EASD + JS':<25} {adaptive.tokens_per_second:>8.1f} {adaptive.acceptance_rate:>7.1%} {prompt_results['configs']['adaptive_full']['speedup']:>7.2f}x")
    
    # ── AGGREGATE RESULTS ──
    print(f"\n\n{'='*60}")
    print(f"  AGGREGATE BENCHMARK RESULTS ({len(PROMPTS)} prompts)")
    print(f"{'='*60}")
    
    for config_name in ["fixed_k5", "adaptive_no_easd", "adaptive_full"]:
        speedups = [r["configs"][config_name]["speedup"] for r in results]
        accept_rates = [r["configs"][config_name]["acceptance_rate"] for r in results]
        throughputs = [r["configs"][config_name]["throughput"] for r in results]
        
        avg_speedup = sum(speedups) / len(speedups)
        avg_accept = sum(accept_rates) / len(accept_rates)
        avg_throughput = sum(throughputs) / len(throughputs)
        
        print(f"\n  {config_name}:")
        print(f"    Avg speedup:    {avg_speedup:.2f}x")
        print(f"    Avg acceptance: {avg_accept:.1%}")
        print(f"    Avg throughput: {avg_throughput:.1f} tok/s")
    
    # ── SAVE ──
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Results saved to {save_path}]")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--save", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    run_benchmark(max_tokens=args.max_tokens, save_path=args.save)