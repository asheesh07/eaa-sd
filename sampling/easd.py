import math
import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
from utils.caching import prune_cache
from typing import List, Tuple


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def max_fn(x: torch.Tensor) -> torch.Tensor:
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


def _entropy(probs: torch.Tensor) -> float:
    return -(probs * torch.log(probs + 1e-10)).sum().item()


def _js_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log2(p + 1e-10) - torch.log2(m + 1e-10))).sum().item()
    kl_qm = (q * (torch.log2(q + 1e-10) - torch.log2(m + 1e-10))).sum().item()
    return math.sqrt(max(0.0, 0.5 * kl_pm + 0.5 * kl_qm))


def _adaptive_k(entropy: float, max_gamma: int) -> int:
    if entropy < 1.0:
        k = 8
    elif entropy < 2.0:
        k = 5
    elif entropy < 3.0:
        k = 3
    else:
        k = 1
    return min(k, max_gamma)


# ─────────────────────────────────────────────
# Adaptive Threshold Tracker
# ─────────────────────────────────────────────

class _AdaThresholds:

    def __init__(self, min_obs: int = 5):
        self._min = min_obs
        self._rej_H = []
        self._acc_js = []
        self._rej_js = []

    def record_accepted(self, js: float):
        self._acc_js.append(js)

    def record_rejected(self, H: float, js: float):
        self._rej_H.append(H)
        self._rej_js.append(js)

    @property
    def TG(self):
        if len(self._rej_H) < self._min:
            return float("inf")
        return sum(self._rej_H) / len(self._rej_H)

    @property
    def TV(self):
        if len(self._acc_js) < self._min or len(self._rej_js) < self._min:
            return -1.0
        return (
            sum(self._acc_js) / len(self._acc_js)
            + sum(self._rej_js) / len(self._rej_js)
        ) / 2.0

    def summary(self):
        return {
            "TG": self.TG,
            "TV": self.TV,
            "mean_acc_js": sum(self._acc_js) / len(self._acc_js) if self._acc_js else 0.0,
            "mean_rej_js": sum(self._rej_js) / len(self._rej_js) if self._rej_js else 0.0,
            "n_acc": len(self._acc_js),
            "n_rej": len(self._rej_H),
        }


# ─────────────────────────────────────────────
# EASD (Incremental Version)
# ─────────────────────────────────────────────

@torch.no_grad()
def easd_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    max_gamma: int = 10,
    ada_min_obs: int = 5,
    use_tv_override: bool = True,
    logits_processor: LogitsProcessor = None,
    max_gen_len: int = 200,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = True,
    debug: bool = False,
) -> Tuple[List[int], dict]:

    if logits_processor is None:
        logits_processor = GreedyProcessor()

    drafter_cache = None
    target_cache = None

    ada = _AdaThresholds(min_obs=ada_min_obs)

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, device=target.device).unsqueeze(1)

    prompt_len = len(inputs)
    total_len = prompt_len + max_gen_len

    input_ids = torch.full(
        (1, total_len),
        pad_token_id,
        dtype=torch.long,
        device=target.device,
    )
    input_ids[0, :prompt_len] = torch.tensor(inputs, device=target.device)

    current_position = prompt_len

    # ── Prefill target ─────────────────────────────
    Mp = target(
        input_ids=input_ids[..., :current_position],
        use_cache=True,
    )
    target_cache = Mp.past_key_values

    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, current_position] = t
    current_position += 1

    drafts_accepted = 0.0
    drafts_speculated = 0.0
    tv_overrides = 0
    k_values = []

    # ── Main Loop ─────────────────────────────
    while current_position < total_len:

        max_this_round = min(max_gamma, total_len - current_position - 1)
        if max_this_round <= 0:
            break

        q_list = []
        entropy_list = []
        actual_gamma = 0

        input_ids = input_ids.to(drafter.device)

        # ── Draft loop ─────────────────────────
        for k in range(max_this_round):

            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=True,
            )
            drafter_cache = Mq.past_key_values

            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            probs_1d = draft_probs[0]

            H = _entropy(probs_1d)
            entropy_list.append(H)
            q_list.append(probs_1d.to(target.device))

            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
            actual_gamma += 1

            if H > ada.TG:
                break

            if k + 1 >= _adaptive_k(H, max_this_round):
                break

        q = torch.stack(q_list, dim=0).unsqueeze(0)
        drafts_speculated += actual_gamma
        k_values.append(actual_gamma)

        input_ids = input_ids.to(target.device)

        # ── Incremental Target Verification ─────────────
        draft_tokens = input_ids[
            ..., current_position: current_position + actual_gamma
        ]

        Mp = target(
            input_ids=draft_tokens,
            past_key_values=target_cache,
            use_cache=True,
        )

        target_cache_full = Mp.past_key_values
        p = logits_processor(Mp.logits)

        r = torch.rand(actual_gamma, device=target.device)
        fractions = p / (q + 1e-10)
        n = actual_gamma
        TV = ada.TV

        for i in range(actual_gamma):

            draft_tok = draft_tokens[0, i]
            p_i = p[0, i]
            q_i = q[0, i]

            js = _js_distance(p_i, q_i)

            std_accept = r[i] <= fractions[0, i, draft_tok]
            tv_accept = use_tv_override and (TV >= 0.0) and (js < TV)

            if std_accept or tv_accept:
                drafts_accepted += 1
                ada.record_accepted(js)
                if tv_accept and not std_accept:
                    tv_overrides += 1
            else:
                ada.record_rejected(entropy_list[i], js)
                n = i
                break

        # ── Cache correction ─────────────
        if n < actual_gamma:
            drafter_cache = prune_cache(drafter_cache, actual_gamma - n)
            target_cache = prune_cache(target_cache_full, actual_gamma - n)
        else:
            target_cache = target_cache_full

        # ── Bonus token ─────────────
        if n == actual_gamma:
            p_p = logits_processor(Mp.logits[..., -1, :])
        else:
            p_p = max_fn(p[..., n, :] - q[0, n, :])

        x = logits_processor.sample(p_p)

        input_ids[0, current_position + n] = x
        current_position += n + 1

        if torch.isin(x, stop_tokens):
            break

    stats = {
        "acceptance_rate": drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0.0,
        "drafts_accepted": int(drafts_accepted),
        "drafts_speculated": int(drafts_speculated),
        "verification_rounds": len(k_values),
        "avg_k": sum(k_values) / len(k_values) if k_values else 0.0,
        "k_min": min(k_values) if k_values else 0,
        "k_max": max(k_values) if k_values else 0,
        "tv_overrides": tv_overrides,
        **ada.summary(),
    }

    return input_ids[0, prompt_len:current_position].tolist(), stats