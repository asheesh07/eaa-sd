# easd.py  — Entropy-Aware Adaptive Speculative Decoding
#
# Same structure as speculative_generate() in sampling.py.
# Three changes only:
#   1. gamma computed per-token from entropy        (adaptive K)
#   2. draft stops early when entropy > TG          (generation threshold)
#   3. token accepted via JS distance when TV met   (verification threshold)
#
# Both TG and TV update automatically from running stats — no hyperparameters.

import math
import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def max_fn(x: torch.Tensor) -> torch.Tensor:
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


def _entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a 1-D probability vector (already softmaxed)."""
    return -(probs * torch.log(probs + 1e-10)).sum().item()


def _js_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Jensen-Shannon distance = sqrt(JS divergence), base-2 log.
    p = target probs [vocab], q = draft probs [vocab].
    Always in [0, 1].
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log2(p + 1e-10) - torch.log2(m + 1e-10))).sum().item()
    kl_qm = (q * (torch.log2(q + 1e-10) - torch.log2(m + 1e-10))).sum().item()
    return math.sqrt(max(0.0, 0.5 * kl_pm + 0.5 * kl_qm))


def _adaptive_k(entropy: float, max_gamma: int) -> int:
    """
    How many tokens to draft given current entropy.
    Thresholds tuned for Qwen 2.5 (entropy range roughly 0-5).
    Adjust after running calibrate_entropy() in run_benchmark.py.

        entropy < 1.0  -> very confident  -> draft up to 8 tokens
        entropy < 2.0  -> confident       -> draft up to 5 tokens
        entropy < 3.0  -> uncertain       -> draft up to 3 tokens
        entropy >= 3.0 -> very uncertain  -> draft 1 token
    """
    if entropy < 1.0:
        k = 8
    elif entropy < 2.0:
        k = 5
    elif entropy < 3.0:
        k = 3
    else:
        k = 1
    return min(k, max_gamma)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive threshold tracker  (AdaSD-style, arXiv:2512.11280)
# ─────────────────────────────────────────────────────────────────────────────

class _AdaThresholds:
    """
    TG = mean entropy of previously rejected tokens
         -> stop drafting when H(current token) > TG

    TV = (mean_JS_accepted + mean_JS_rejected) / 2
         -> accept via JS distance even when sampled tokens differ

    Both disabled (inf / -1) until min_obs observations of each type.
    """

    def __init__(self, min_obs: int = 5):
        self._min     = min_obs
        self._rej_H:  List[float] = []
        self._acc_js: List[float] = []
        self._rej_js: List[float] = []

    def record_accepted(self, js: float):
        self._acc_js.append(js)

    def record_rejected(self, H: float, js: float):
        self._rej_H.append(H)
        self._rej_js.append(js)

    @property
    def TG(self) -> float:
        if len(self._rej_H) < self._min:
            return float("inf")
        return sum(self._rej_H) / len(self._rej_H)

    @property
    def TV(self) -> float:
        if len(self._acc_js) < self._min or len(self._rej_js) < self._min:
            return -1.0
        mean_acc = sum(self._acc_js) / len(self._acc_js)
        mean_rej = sum(self._rej_js) / len(self._rej_js)
        return (mean_acc + mean_rej) / 2.0

    def summary(self) -> dict:
        return {
            "TG":          self.TG,
            "TV":          self.TV,
            "mean_acc_js": (sum(self._acc_js) / len(self._acc_js)) if self._acc_js else 0.0,
            "mean_rej_js": (sum(self._rej_js) / len(self._rej_js)) if self._rej_js else 0.0,
            "n_acc":       len(self._acc_js),
            "n_rej":       len(self._rej_H),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EASD generate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def easd_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    # ── EASD params ───────────────────────────────────────────────────────────
    max_gamma: int = 10,           # hard ceiling on draft tokens per round
    ada_min_obs: int = 5,          # observations before TG / TV activate
    use_tv_override: bool = True,  # accept via JS distance (TV) when close
    # ── same as speculative_generate ─────────────────────────────────────────
    logits_processor: LogitsProcessor = None,
    max_gen_len: int = 200,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = True,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], dict]:
    """
    Entropy-Aware Adaptive Speculative Decoding.

    Returns:
        output_ids : List[int]  generated token ids (excluding prompt)
        stats      : dict       benchmark metrics

    Usage:
        output_ids, stats = easd_generate(prompt_ids, drafter, target, ...)
        print_stats(stats)
    """
    if logits_processor is None:
        logits_processor = GreedyProcessor()

    drafter_cache, target_cache = None, None

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(
        list_tokens_id, dtype=torch.long, device=target.device
    ).unsqueeze(1)

    drafts_accepted   = 0.0
    drafts_speculated = 0.0
    tv_overrides      = 0
    k_values: List[int] = []

    ada = _AdaThresholds(min_obs=ada_min_obs)
    vocabulary_size = target.config.vocab_size

    # ── buffer setup (identical to speculative_generate) ─────────────────────
    prompt_len = len(inputs)
    max_seq_length = (
        target.config.max_position_embeddings
        if hasattr(target.config, "max_position_embeddings")
        else (target.config.max_context_length
              if hasattr(target.config, "max_context_length")
              else 1024)
    )
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=target.device
    )
    input_ids[0, :prompt_len] = torch.tensor(
        inputs, dtype=torch.long, device=target.device
    )
    current_position = prompt_len

    # ── first target pass (identical to speculative_generate) ─────────────────
    if first_target:
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t   = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1

        if torch.isin(t, stop_tokens):
            return (
                input_ids[0, prompt_len:current_position].tolist(),
                _make_stats(0, 0, [], 0, ada),
            )
        if debug:
            printing.initial_step(t, tokenizer)

    # ── main loop ─────────────────────────────────────────────────────────────
    while current_position < total_len:

        max_this_round = min(max_gamma, total_len - current_position - 1)
        if max_this_round <= 0:
            break

        # ── CHANGE 1: draft one token at a time, check entropy after each ─────
        q_list:       List[torch.Tensor] = []
        entropy_list: List[float]        = []
        actual_gamma  = 0

        input_ids = input_ids.to(drafter.device)

        for k in range(max_this_round):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values

            draft_logits = Mq.logits[..., -1, :]
            draft_probs  = logits_processor(draft_logits)   # [1, vocab]
            probs_1d     = draft_probs[0]                   # [vocab]

            H = _entropy(probs_1d)
            entropy_list.append(H)
            q_list.append(probs_1d.to(target.device))

            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
            actual_gamma += 1

            # ── CHANGE 2: TG early stop ───────────────────────────────────────
            TG = ada.TG
            if H > TG:
                if debug:
                    print(f"[EASD] TG stop  k={k}  H={H:.3f} > TG={TG:.3f}")
                break

            # entropy-based K ceiling
            if k + 1 >= _adaptive_k(H, max_this_round):
                break

        # q: [1, actual_gamma, vocab]
        q = torch.stack(q_list, dim=0).unsqueeze(0)
        drafts_speculated += actual_gamma
        k_values.append(actual_gamma)

        input_ids = input_ids.to(target.device)

        # ── target verification ───────────────────────────────────────────────
        Mp = target(
            input_ids=input_ids[..., :current_position + actual_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values

        p = logits_processor(
            Mp.logits[
                ...,
                current_position - 1: current_position + actual_gamma - 1,
                :,
            ]
        )   # [1, actual_gamma, vocab]

        # ── CHANGE 3: acceptance with JS distance override ────────────────────
        r         = torch.rand(actual_gamma, device=target.device)
        fractions = p / (q + 1e-10)
        n         = actual_gamma
        TV        = ada.TV

        for i in range(actual_gamma):
            draft_tok = input_ids[0, current_position + i].item()
            p_i = p[0, i]
            q_i = q[0, i]
            js  = _js_distance(p_i, q_i)

            std_accept = r[i] <= fractions[0, i, draft_tok]
            tv_accept  = use_tv_override and (TV >= 0.0) and (js < TV)

            if std_accept or tv_accept:
                ada.record_accepted(js)
                drafts_accepted += 1
                if tv_accept and not std_accept:
                    tv_overrides += 1
            else:
                ada.record_rejected(H=entropy_list[i], js=js)
                n = i
                break

        # ── cache pruning (identical to speculative_generate) ─────────────────
        if n < actual_gamma and use_cache:
            drafter_cache = prune_cache(drafter_cache, actual_gamma - n)
            target_cache  = prune_cache(target_cache,  actual_gamma - n + 1)

        # ── EOS check ─────────────────────────────────────────────────────────
        stop_locations = torch.nonzero(
            torch.eq(
                input_ids[..., current_position: current_position + n],
                stop_tokens,
            )
        )
        if stop_locations.shape[0] > 0:
            stop_loc = stop_locations[0, 1].item()
            if debug:
                printing.end_token_found(stop_loc)
            return (
                input_ids[0, prompt_len: current_position + stop_loc + 1].tolist(),
                _make_stats(drafts_accepted, drafts_speculated, k_values, tv_overrides, ada),
            )

        # ── bonus token (identical to speculative_generate) ───────────────────
        if n == actual_gamma:
            p_p = logits_processor(
                Mp.logits[..., current_position + actual_gamma - 1, :]
            )
        else:
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]

        x = logits_processor.sample(p_p)

        if debug:
            generated = input_ids.clone().detach()

        input_ids[0, current_position + n: current_position + actual_gamma] = pad_token_id
        input_ids[0, current_position + n] = x

        if debug:
            printing.speculative_step(
                tokenizer, generated, input_ids,
                n, prompt_len, current_position, actual_gamma,
            )

        current_position += n + 1

        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(n)
            return (
                input_ids[0, prompt_len: current_position].tolist(),
                _make_stats(drafts_accepted, drafts_speculated, k_values, tv_overrides, ada),
            )

    return (
        input_ids[0, prompt_len:].tolist(),
        _make_stats(drafts_accepted, drafts_speculated, k_values, tv_overrides, ada),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stats(accepted, speculated, k_values, tv_overrides, ada) -> dict:
    return {
        "acceptance_rate":     accepted / speculated if speculated > 0 else 0.0,
        "drafts_accepted":     int(accepted),
        "drafts_speculated":   int(speculated),
        "verification_rounds": len(k_values),
        "avg_k":               sum(k_values) / len(k_values) if k_values else 0.0,
        "k_min":               min(k_values) if k_values else 0,
        "k_max":               max(k_values) if k_values else 0,
        "tv_overrides":        tv_overrides,
        **ada.summary(),
    }


def print_stats(stats: dict):
    print("  ─────────────────────────────────────────")
    print(f"  Acceptance rate:     {stats['acceptance_rate']:.1%}")
    print(f"  Accepted / Drafted:  {stats['drafts_accepted']} / {stats['drafts_speculated']}")
    print(f"  Verification rounds: {stats['verification_rounds']}")
    print(f"  Avg adaptive K:      {stats['avg_k']:.2f}")
    print(f"  K range:             [{stats['k_min']}, {stats['k_max']}]")
    print(f"  TV overrides:        {stats['tv_overrides']}")
    print(f"  TG (final):          {stats['TG']:.4f}")
    print(f"  TV (final):          {stats['TV']:.4f}")
    print(f"  Mean JS accepted:    {stats['mean_acc_js']:.4f}")
    print(f"  Mean JS rejected:    {stats['mean_rej_js']:.4f}")
    print("  ─────────────────────────────────────────")