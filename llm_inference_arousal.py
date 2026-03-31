import os
import re
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
os.environ["VLLM_USE_V1"] = "0"
# ----------------------------
# CONFIG
# ----------------------------
BASE_MODEL = "Qwen/Qwen3-8B"
LORA_DIR = "./qwen3_8b_ordinal_lora_arousal_v2"   # <- your trained adapter dir
LORA_NAME = "arousal_lora"
LORA_ID = 1

MAX_LENGTH = 384
BATCH_SIZE = 128  # bump up/down based on memory + throughput

DIGIT_MIN = 3
DIGIT_MAX = 8
DIGITS = list(range(DIGIT_MIN, DIGIT_MAX + 1))

# If digits sometimes don't appear in top_logprobs, increase this.
# vLLM returns top-K logprobs per output token when logprobs is set. :contentReference[oaicite:2]{index=2}
NUM_LOGPROBS = 20


def build_prompt(text, aspect, language, domain):
    return f"""You are a sentiment classification system.

Task:
Given a text and an aspect, output a single integer from {DIGIT_MIN} to
{DIGIT_MAX} representing the arousal expressed toward the aspect.

Output rules:
- Output exactly ONE integer
- No explanation
- No additional text

Language: {language}
Domain: {domain}
Text: {text}
Aspect: {aspect}

Output: """

# ----------------------------
# Digit token ids (must match tokenizer)
# ----------------------------
def get_digit_token_ids(tokenizer):
    # Assert each digit is a single token (true for most BPE tokenizers, but verify).
    digit_token_ids = {}
    for d in DIGITS:
        ids = tokenizer(str(d), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Digit '{d}' is not a single token under this tokenizer: {ids}")
        digit_token_ids[d] = ids[0]
    return digit_token_ids

def extract_digit_probs_from_output(out, digit_token_ids_list):
    """
    Returns probs in DIGITS order [3..8].
    Assumes max_tokens=1.
    Works with vLLM where out.outputs[0].logprobs is a list of dicts (per position).
    """
    if (not out.outputs) or (out.outputs[0].logprobs is None) or (len(out.outputs[0].logprobs) == 0):
        return None

    pos0 = out.outputs[0].logprobs[0]  # dict[token_id] -> Logprob-like OR float
    lps = []
    for tid in digit_token_ids_list:
        if tid in pos0:
            v = pos0[tid]
            lp = v.logprob if hasattr(v, "logprob") else float(v)
        else:
            lp = float("-inf")
        lps.append(lp)

    m = max(lps)
    exps = [0.0 if lp == float("-inf") else math.exp(lp - m) for lp in lps]
    Z = sum(exps)
    if Z == 0:
        # ultra-defensive fallback
        return [1.0 / len(digit_token_ids_list)] * len(digit_token_ids_list)
    return [e / Z for e in exps]


def summarize_probs(probs):
    probs = list(probs)
    parts = []
    for d, p in zip(DIGITS, probs):
        try:
            p = float(p)
        except Exception:
            p = 0.0
        parts.append(f"{d}={p:.3f}")
    return " ".join(parts)

# ----------------------------
# Main inference
# ----------------------------
def infer_guidance(df, llm, tokenizer, lora_request, out_path_csv=None, out_path_parquet=None):
    digit_token_ids = get_digit_token_ids(tokenizer)
    digit_token_ids_list = [digit_token_ids[d] for d in DIGITS]  # token ids in DIGITS order

    # logits processor closure (captures digit_token_ids_list)
    def only_digit_logits_processor(*args):
        logits = args[-1]  # tensor [batch, vocab]
        masked = torch.full_like(logits, float("-inf"))
        masked[:, digit_token_ids_list] = logits[:, digit_token_ids_list]
        return masked

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=NUM_LOGPROBS,              # vLLM cap is fine
        allowed_token_ids=digit_token_ids_list,  # ✅ V1-supported way to restrict tokens
        stop=["\n", "</s>", "<|im_end|>"],
    )

    rows = []
    prompts = []
    meta = []

    # Build prompts
    for r in df.itertuples(index=False):
        prompts.append(build_prompt(r.Text, r.Aspect, r.Language, r.Domain))
        meta.append((r.ID, r.Language, r.Domain, r.Aspect))

    # Batch generate
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="vLLM inference"):
        p_batch = prompts[i:i+BATCH_SIZE]
        m_batch = meta[i:i+BATCH_SIZE]

        outputs = llm.generate(
            p_batch,
            sampling_params,
            lora_request=lora_request,
        )

        for out, (ex_id, lang, domain, aspect) in zip(outputs, m_batch):
            gen_text = out.outputs[0].text.strip() if out.outputs else ""

            # FIX A: in your vLLM, logprobs is a list (per generated token position)
            logprobs_list = out.outputs[0].logprobs if (out.outputs and out.outputs[0].logprobs is not None) else None

            if not logprobs_list or len(logprobs_list) == 0:
                # fallback: parse the generated digit
                m = re.search(r"[0-9]", gen_text)
                hard_digit = int(m.group(0)) if m else 6
                hard_digit = int(np.clip(hard_digit, DIGIT_MIN, DIGIT_MAX))
                probs = np.zeros(len(DIGITS), dtype=np.float64)
                probs[DIGITS.index(hard_digit)] = 1.0
            else:
                # Use your helper (expects global DIGIT_TOKEN_IDS, so set it here)
                global DIGIT_TOKEN_IDS
                DIGIT_TOKEN_IDS = digit_token_ids_list  # aligns with DIGITS
                probs = extract_digit_probs_from_output(out, digit_token_ids_list)

                if probs is None:
                    m = re.search(r"[0-9]", gen_text)
                    hard_digit = int(m.group(0)) if m else 6
                    hard_digit = int(np.clip(hard_digit, DIGIT_MIN, DIGIT_MAX))
                    probs = np.zeros(len(DIGITS), dtype=np.float64)
                    probs[DIGITS.index(hard_digit)] = 1.0
                else:
                    probs = np.array(probs, dtype=np.float64)
                    hard_digit = int(DIGITS[int(np.argmax(probs))])

            # Expected value + p_low
            p_low = float(probs[0:3].sum())  # P(3,4,5)
            exp_score = float(np.dot(np.array(DIGITS, dtype=np.float64), probs))
            hard_bin = "[BIN_B]" if hard_digit < 6 else "[BIN_C]"

            rows.append({
                "ID": ex_id,
                "Language": lang,
                "Domain": domain,
                "Aspect": aspect,
                "llm_digit": hard_digit,
                "llm_bin_bc": hard_bin,
                "llm_p_low": p_low,
                "llm_exp_score": exp_score,
                "llm_dist_str": summarize_probs(probs),
            })

    out_df = pd.DataFrame(rows)

    if out_path_csv:
        os.makedirs(os.path.dirname(out_path_csv), exist_ok=True)
        out_df.to_csv(out_path_csv, index=False)
    if out_path_parquet:
        os.makedirs(os.path.dirname(out_path_parquet), exist_ok=True)
        out_df.to_parquet(out_path_parquet, index=False)

    return out_df


# ----------------------------
# Example usage (expects you already have train/dev/test dataframes)
# ----------------------------
if __name__ == "__main__":
    # Load tokenizer (for digit token ids)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load vLLM engine
    llm = LLM(
        model=BASE_MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,          # or higher if you’ll use multiple adapters
        max_lora_rank=64,     # must be >= your trained LoRA r (yours was r=16)
    )

    lora_request = LoRARequest(LORA_NAME, LORA_ID, LORA_DIR)

    # -----
    # You should provide these dataframes from your existing download/normalize code:
    # train_source_df, official_dev_df, official_test_df
    # For demo, we'll assume you saved them previously as parquet/csv and reload here.
    # -----
    train_source_df = pd.read_csv("github_datasets/train_source_df.csv")
    official_dev_df = pd.read_csv("github_datasets/official_dev_df.csv")
    official_test_df = pd.read_csv("github_datasets/official_test_df.csv")

    # Ensure required columns exist
    required = ["ID", "Text", "Aspect", "Language", "Domain"]
    # (dev/test have no GT, that's fine)
    for name, df_ in [("train_source_df", train_source_df),
                      ("official_dev_df", official_dev_df),
                      ("official_test_df", official_test_df)]:
        for c in required:
            if c not in df_.columns:
                raise ValueError(f"{name} missing column {c}")

    # Run inference + save
    train_guidance = infer_guidance(
        train_source_df,
        llm, tokenizer, lora_request,
        out_path_csv="llm_preds_final_v3/arousal_guidance_train.csv",
    )

    dev_guidance = infer_guidance(
        official_dev_df,
        llm, tokenizer, lora_request,
        out_path_csv="llm_preds_final_v3/arousal_guidance_dev.csv",
    )

    test_guidance = infer_guidance(
        official_test_df,
        llm, tokenizer, lora_request,
        out_path_csv="llm_preds_final_v3/arousal_guidance_test.csv",
    )

    print("Done.")
