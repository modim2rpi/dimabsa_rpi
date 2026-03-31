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

os.environ["VLLM_USE_V1"] = "0"

# ----------------------------
# CONFIG
# ----------------------------
# If you MERGED the model + saved it, set BASE_MODEL to that directory and USE_LORA=False.
# If you did NOT merge and still want LoRA inference, set BASE_MODEL to Qwen/Qwen3-8B and USE_LORA=True.
USE_LORA = False

BASE_MODEL = "Qwen/Qwen3-8B"
# BASE_MODEL = "./qwen3_8b_ordinal_lora_valence_v3/merged"  # example merged dir (then USE_LORA=False)

LORA_DIR = "./qwen3_8b_ordinal_lora_valence_v3/merged_full_model"  # your adapter dir (tokenizer likely here too)
LORA_NAME = "valence_lora"
LORA_ID = 1

OUT_DIR = "llm_preds_final_v3"
# os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH = 384
BATCH_SIZE = 128
NUM_LOGPROBS = 20

DIGIT_MIN = 1
DIGIT_MAX = 8
DIGITS = list(range(DIGIT_MIN, DIGIT_MAX + 1))

# If your tokenizer has <ANS> (recommended), keep this consistent with training prompt.
ANS_TOKEN = "<ANS>"


def build_prompt(text, aspect, language, domain):
    # NOTE: We keep Output: <ANS> at the end if that was your training format.
    # If you used a normal prompt without <ANS>, remove it here.
    return f"""You are a sentiment classification system.

Task:
Given a text and an aspect, output a single integer from {DIGIT_MIN} to {DIGIT_MAX} representing the valence expressed toward the aspect.

Output rules:
- Output exactly ONE integer
- No explanation
- No additional text

Language: {language}
Domain: {domain}
Text: {text}
Aspect: {aspect}

Output: {ANS_TOKEN}"""


# ----------------------------
# Token id helpers
# ----------------------------
def get_digit_token_ids(tokenizer):
    digit_token_ids = {}
    for d in DIGITS:
        ids = tokenizer(str(d), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Digit '{d}' is not a single token under this tokenizer: {ids}")
        digit_token_ids[d] = ids[0]
    return digit_token_ids


def extract_digit_probs_from_output(out, digit_token_ids_list):
    """
    Returns probs in DIGITS order [1..8].
    Assumes max_tokens=1.
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
        return [1.0 / len(digit_token_ids_list)] * len(digit_token_ids_list)
    return [e / Z for e in exps]


def summarize_probs(probs):
    parts = []
    for d, p in zip(DIGITS, probs):
        parts.append(f"{d}={float(p):.3f}")
    return " ".join(parts)


def bin_from_digit(d):
    if d < 3:
        return "[BIN_A]"
    elif d < 6:
        return "[BIN_B]"
    else:
        return "[BIN_C]"


def compute_bin_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    # A: 1,2
    pA = float(probs[0:2].sum())
    # B: 3,4,5
    pB = float(probs[2:5].sum())
    # C: 6,7,8
    pC = float(probs[5:8].sum())
    return pA, pB, pC


# ----------------------------
# Main inference
# ----------------------------
def infer_guidance(df, llm, tokenizer, lora_request=None, out_path_csv=None):
    digit_token_ids = get_digit_token_ids(tokenizer)
    digit_token_ids_list = [digit_token_ids[d] for d in DIGITS]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=NUM_LOGPROBS,
        allowed_token_ids=digit_token_ids_list,
        stop=["\n", "</s>", "<|im_end|>"],
    )

    prompts = []
    meta = []
    for r in df.itertuples(index=False):
        prompts.append(build_prompt(r.Text, r.Aspect, r.Language, r.Domain))
        meta.append((r.ID, r.Language, r.Domain, r.Aspect))

    rows = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="vLLM valence inference"):
        p_batch = prompts[i:i+BATCH_SIZE]
        m_batch = meta[i:i+BATCH_SIZE]

        outputs = llm.generate(p_batch, sampling_params, lora_request=lora_request)

        for out, (ex_id, lang, domain, aspect) in zip(outputs, m_batch):
            gen_text = out.outputs[0].text.strip() if out.outputs else ""

            probs = extract_digit_probs_from_output(out, digit_token_ids_list)
            if probs is None:
                # fallback: parse a digit from generated text
                m = re.search(r"[1-8]", gen_text)
                hard_digit = int(m.group(0)) if m else 5
                hard_digit = int(np.clip(hard_digit, DIGIT_MIN, DIGIT_MAX))
                probs = np.zeros(len(DIGITS), dtype=np.float64)
                probs[DIGITS.index(hard_digit)] = 1.0
            else:
                probs = np.array(probs, dtype=np.float64)
                hard_digit = int(DIGITS[int(np.argmax(probs))])

            exp_score = float(np.dot(np.array(DIGITS, dtype=np.float64), probs))
            conf = float(np.max(probs))
            pA, pB, pC = compute_bin_probs(probs)
            hard_bin = bin_from_digit(hard_digit)

            rows.append({
                "ID": ex_id,
                "Language": lang,
                "Domain": domain,
                "Aspect": aspect,
                "llm_digit": hard_digit,
                "llm_bin_abc": hard_bin,
                "llm_pA": pA,
                "llm_pB": pB,
                "llm_pC": pC,
                "llm_conf": conf,
                "llm_exp_score": exp_score,
                "llm_dist_str": summarize_probs(probs),
            })

    out_df = pd.DataFrame(rows)

    if out_path_csv:
        os.makedirs(os.path.dirname(out_path_csv), exist_ok=True)
        out_df.to_csv(out_path_csv, index=False)

    return out_df


if __name__ == "__main__":
    # Load dataframes saved from your normalize script
    train_source_df = pd.read_csv("github_datasets/train_source_df.csv")
    official_dev_df = pd.read_csv("github_datasets/official_dev_df.csv")
    official_test_df = pd.read_csv("github_datasets/official_test_df.csv")

    required = ["ID", "Text", "Aspect", "Language", "Domain"]
    for name, df_ in [("train_source_df", train_source_df),
                      ("official_dev_df", official_dev_df),
                      ("official_test_df", official_test_df)]:
        for c in required:
            if c not in df_.columns:
                raise ValueError(f"{name} missing column {c}")

    # Tokenizer: if you trained with <ANS> as a new token, you must load the tokenizer that contains it.
    # Usually it’s in your training output dir.
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)

    # vLLM engine
    llm = LLM(
        model=BASE_MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=False,
        max_loras=1,
        max_lora_rank=64,
    )

    lora_request = None
    if USE_LORA:
        lora_request = LoRARequest(LORA_NAME, LORA_ID, LORA_DIR)

    train_guidance = infer_guidance(
        train_source_df, llm, tokenizer, lora_request=lora_request,
        out_path_csv=os.path.join(OUT_DIR, "valence_guidance_train.csv"),
    )
    dev_guidance = infer_guidance(
        official_dev_df, llm, tokenizer, lora_request=lora_request,
        out_path_csv=os.path.join(OUT_DIR, "valence_guidance_dev.csv"),
    )
    test_guidance = infer_guidance(
        official_test_df, llm, tokenizer, lora_request=lora_request,
        out_path_csv=os.path.join(OUT_DIR, "valence_guidance_test.csv"),
    )

    print("Done.")