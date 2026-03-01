"""
Run — Entry point for Entropy-Aware Adaptive-K Speculative Decoding
=====================================================================

Usage:
    python run.py --prompt "Explain quantum computing" --max_tokens 256
    python run.py --prompt "Write a poem about stars" --benchmark
    python run.py --prompt "What is machine learning?" --no-easd --fixed-k 5
"""

import argparse
import torch
import json
from config import SpeculativeConfig, ModelConfig, EntropyConfig, GenerationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-Aware Adaptive-K Speculative Decoding"
    )
    
    # Core args
    parser.add_argument("--prompt", type=str, 
                        default="Explain the theory of general relativity in simple terms.",
                        help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    
    # Model args
    parser.add_argument("--draft_model", type=str, 
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--target_model", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization for target model")
    
    # Entropy args
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=12)
    parser.add_argument("--k_initial", type=int, default=5)
    parser.add_argument("--fixed-k", type=int, default=None,
                        help="Use fixed K instead of adaptive (for ablation)")
    parser.add_argument("--entropy_low", type=float, default=0.5)
    parser.add_argument("--entropy_high", type=float, default=2.5)
    
    # Feature flags
    parser.add_argument("--no-easd", action="store_true",
                        help="Disable EASD entropy penalty")
    parser.add_argument("--no-js", action="store_true",
                        help="Disable JS divergence adaptive threshold")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")
    
    # Benchmark
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing speculative vs vanilla")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_stats", type=str, default=None,
                        help="Save stats to JSON file")
    
    args = parser.parse_args()
    
    # ── Build config ──
    config = SpeculativeConfig(
        model=ModelConfig(
            draft_model_name=args.draft_model,
            target_model_name=args.target_model,
            use_4bit_target=not args.no_4bit,
        ),
        entropy=EntropyConfig(
            k_min=args.k_min,
            k_max=args.k_max,
            k_initial=args.k_initial,
            entropy_low=args.entropy_low,
            entropy_high=args.entropy_high,
            easd_enabled=not args.no_easd,
            js_divergence_enabled=not args.no_js,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=not args.greedy,
        ),
        verbose=not args.quiet,
    )
    
    # ── Initialize engine ──
    from speculative_engine import SpeculativeEngine
    engine = SpeculativeEngine(config)
    
    # ── Generate ──
    print(f"\n{'─'*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'─'*60}\n")
    
    output = engine.generate(args.prompt, args.max_tokens)
    
    print(f"\n{'─'*60}")
    print(f"Generated text:")
    print(f"{'─'*60}")
    print(output.text)
    print(f"{'─'*60}\n")
    
    # ── Benchmark (optional) ──
    if args.benchmark:
        print("\n[Benchmark] Running vanilla autoregressive baseline...")
        
        # Reset models
        engine.target_model.reset_cache()
        
        vanilla_output = engine.generate_vanilla(args.prompt, args.max_tokens)
        
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Speculative decoding:")
        print(f"    Throughput: {output.tokens_per_second:.1f} tok/s")
        print(f"    Time:       {output.wall_time_seconds:.2f}s")
        print(f"  Vanilla autoregressive:")
        print(f"    Throughput: {vanilla_output.tokens_per_second:.1f} tok/s")
        print(f"    Time:       {vanilla_output.wall_time_seconds:.2f}s")
        print(f"  ─────────────────────────────────────")
        
        speedup = vanilla_output.wall_time_seconds / max(output.wall_time_seconds, 1e-6)
        print(f"  SPEEDUP: {speedup:.2f}x")
        print(f"  Acceptance rate: {output.acceptance_rate:.1%}")
        print(f"  Avg adaptive K: {output.avg_adaptive_k:.1f}")
        print(f"{'='*60}")
    
    # ── Save stats (optional) ──
    if args.save_stats:
        stats = {
            "prompt": args.prompt,
            "config": {
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "k_initial": args.k_initial,
                "temperature": args.temperature,
                "easd_enabled": not args.no_easd,
                "js_enabled": not args.no_js,
            },
            "results": {
                "total_tokens": output.total_tokens,
                "wall_time": output.wall_time_seconds,
                "throughput": output.tokens_per_second,
                "acceptance_rate": output.acceptance_rate,
                "avg_k": output.avg_adaptive_k,
                "verification_rounds": output.verification_rounds,
                "k_history": output.k_history,
                "entropy_history": output.entropy_history[:50],  # Truncate for readability
            }
        }
        
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n[Stats saved to {args.save_stats}]")


if __name__ == "__main__":
    main()