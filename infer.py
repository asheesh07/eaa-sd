import argparse
import random
import numpy as np
import torch
import time
import os

from sampling import autoregressive_generate
from sampling.easd import easd_generate, print_stats

from ngram_assisted import (
    OneLevelNGramStorage,
    NGramStorage,
    ngram_assisted_speculative_generate,
)

from utils.logits_processor import (
    GreedyProcessor,
    MultinomialProcessor,
    TopKProcessor,
    NucleusProcessor,
    TopKNucleusProcessor,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)

from termcolor import colored


class InferenceCLI:

    def __init__(self, device: str = "cuda"):

        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )

        self.device = device

        self.gen_len = 35
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = True
        self.target_gen = True

        self.ngram_gen = False
        self.ngram = None
        self.top_k_filler = 3
        self.ngram_n = 3
        self.reset_in_between = True

        self.chat = True

        self.processors = {
            "greedy": {
                "processor": GreedyProcessor,
                "building_args": {"temperature": float},
            },
            "multinomial": {
                "processor": MultinomialProcessor,
                "building_args": {"temperature": float},
            },
            "topk": {
                "processor": TopKProcessor,
                "building_args": {"temperature": float, "top_k": int},
            },
            "nucleus": {
                "processor": NucleusProcessor,
                "building_args": {"temperature": float, "top_p": float},
            },
            "topknucleus": {
                "processor": TopKNucleusProcessor,
                "building_args": {
                    "temperature": float,
                    "top_k": int,
                    "top_p": float,
                },
            },
        }

        self.selected_processor = {
            "name": "greedy",
            "processor": GreedyProcessor,
            "args": {"temperature": 1.0},
        }

        self.processor = GreedyProcessor()

        self._load_models()
        self._run()

    # ---------------------------------------------------
    # Model Loading
    # ---------------------------------------------------

    def _load_models(self):

        target_model = "meta-llama/Llama-3.2-3B-Instruct"
        drafter_model = "meta-llama/Llama-3.2-1B-Instruct"

        target_quantize = QuantoConfig(weights="int8")
        drafter_quantize = QuantoConfig(weights="int8")

        print(colored("Target model:", on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=target_quantize,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            trust_remote_code=True,
        )

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=drafter_quantize,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.ngram = NGramStorage(
            n=3,
            vocab_size=self.target.config.vocab_size,
        )

        self.end_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    # ---------------------------------------------------
    # Inference
    # ---------------------------------------------------

    def _infer(self, prefix: str):

        if self.chat:
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prefix}],
                add_generation_prompt=True,
                tokenize=False,
            )

        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()

        if self.reset_in_between:
            self.ngram.reset()

        # ==================================================
        # EASD
        # ==================================================
        if self.spec:

            self._set_seed(42)

            start = time.time()

            output_ids, stats = easd_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                max_gamma=10,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                use_cache=self.cache,
                debug=self.debug,
            )

            end = time.time()

            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            throughput = len(output_ids) / (end - start)

            print(colored("========== EASD ==========", "green"))
            print(output)
            print_stats(stats)
            print(colored(f"Throughput: {throughput:.2f} tokens/s", "green"))
            print(colored("=================================", "green"))

        # ==================================================
        # Target AR Baseline
        # ==================================================
        if self.target_gen:

            self._set_seed(42)

            start = time.time()

            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )

            end = time.time()

            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            throughput = len(output_ids) / (end - start)

            print(colored("=========== Target AR ===========", "blue"))
            print(output)
            print(colored(f"Throughput: {throughput:.2f} tokens/s", "blue"))
            print(colored("=================================", "blue"))

    # ---------------------------------------------------
    # Run Loop
    # ---------------------------------------------------

    def _run(self):

        while True:
            command = input("> ").replace("\\n", "\n").replace("\\t", "\t")

            if command.startswith("/"):
                if command == "/quit":
                    print(colored("Goodbye!", on_color="on_red"))
                    exit(0)
                continue

            self._infer(command)

    # ---------------------------------------------------
    # Seed Control
    # ---------------------------------------------------

    def _set_seed(self, seed: int):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================================================
# Entry
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    InferenceCLI(device=args.device)