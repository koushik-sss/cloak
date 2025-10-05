from gzip import FTEXT
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= Config =========
# Local fine-tuned model directory (must contain config.json, model weights, tokenizer files)
FT_MODEL_DIR = (
    Path(
        os.environ.get(
            # "FT_MODEL_DIR", "/Users/saikoushik/Downloads/trainium_stuff/merged_model"
            "FT_MODEL_DIR",
            "/Users/saikoushik/Downloads/attempt2/merged_model",
        )
    )
    .expanduser()
    .resolve()
)

# Vanilla (stock) model from the Hub
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
DEFAULT_DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() in ("1", "true", "yes")

# ========= System prompt =========
PII_SYSTEM_PROMPT = (
    "You are a personally identifiable information (PII) redaction engine. "
    "Given an input string, transform it according to the following specification:\n\n"
    "Mission\n"
    "- Produce output text where every PII span is redacted and replaced with the string [REDACTED]\n"
    "- Redactions must preserve the original text’s structure, spacing, and non-PII semantics.\n\n"
    "PII coverage (non-exhaustive; use judgment)\n"
    "- Personal names, usernames and handles, full addresses, phone numbers, email addresses.\n"
    "- Government IDs (SSN/SIN/etc.), taxpayer IDs, driver licenses, license plates, passport numbers.\n"
    "- Financial identifiers: bank accounts, credit/debit cards, routing numbers, IBAN/BIC.\n"
    "- Medical record numbers, insurance numbers.\n"
    "- Birth dates, exact ages under 18, geo coordinates tied to individuals, IPs, MAC/device IDs, biometric descriptors.\n"
    "- Student, employee, customer, loyalty, or similar unique identifiers.\n\n"
    "Do not redact\n"
    "- General job titles, non-unique organization names, generic locations (e.g., “the city center”), "
    "standalone years, or other non-identifying context.\n"
)

# ========= Device selection =========
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def _assert_local_ft_dir_ok(path: Path):
    if not path.exists() or not path.is_dir():
        print(
            f"[FATAL] FT_MODEL_DIR does not exist or is not a directory: {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    needed = ["config.json", "tokenizer.json"]
    missing = [f for f in needed if not (path / f).exists()]
    if missing:
        print(f"[FATAL] Missing required files in {path}: {missing}", file=sys.stderr)
        print(
            "Expected at least: config.json, tokenizer.json, and model weights (model.safetensors or pytorch_model.bin).",
            file=sys.stderr,
        )
        sys.exit(1)


# ========= Load models =========
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Loading fine-tuned model from: {FT_MODEL_DIR}")
_assert_local_ft_dir_ok(FT_MODEL_DIR)

ft_tokenizer = AutoTokenizer.from_pretrained(
    str(FT_MODEL_DIR),
    local_files_only=True,
    trust_remote_code=True,
)
ft_model = AutoModelForCausalLM.from_pretrained(
    str(FT_MODEL_DIR),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float32 if DEVICE.type == "cpu" else torch.float16,
)
ft_model.to(DEVICE).eval()

print(f"[INFO] Loading base model from Hub: {BASE_MODEL_ID}")
base_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32 if DEVICE.type == "cpu" else torch.float16,
)
base_model.to(DEVICE).eval()


# ========= Helper: prompt formatting =========
def build_prompt(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PII_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback TinyLlama-ish format
    return (
        f"<|system|>\n{PII_SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_text}</s>\n"
        f"<|assistant|>\n"
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ========= FastAPI =========
app = FastAPI(title="PII Redaction Comparison Server")

# Add CORS middleware to allow all origins (including file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class RedactRequest(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = None


def _resolve_params(req: RedactRequest):
    return (
        req.max_new_tokens
        if req.max_new_tokens is not None
        else DEFAULT_MAX_NEW_TOKENS,
        req.do_sample if req.do_sample is not None else DEFAULT_DO_SAMPLE,
        req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE,
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "ft_model_dir": str(FT_MODEL_DIR),
        "base_model_id": BASE_MODEL_ID,
    }


@app.post("/redact")
def redact_ft(req: RedactRequest):
    max_new, do_sample, temp = _resolve_params(req)
    prompt = build_prompt(ft_tokenizer, req.text)
    t0 = time.time()
    out = generate_text(ft_model, ft_tokenizer, prompt, max_new, do_sample, temp)
    dt = time.time() - t0
    return {"model": "fine_tuned", "latency_s": round(dt, 3), "redacted": out}


@app.post("/redact_base")
def redact_base(req: RedactRequest):
    max_new, do_sample, temp = _resolve_params(req)
    prompt = build_prompt(base_tokenizer, req.text)
    t0 = time.time()
    out = generate_text(base_model, base_tokenizer, prompt, max_new, do_sample, temp)
    dt = time.time() - t0
    return {"model": "base", "latency_s": round(dt, 3), "redacted": out}


@app.post("/compare")
def compare(req: RedactRequest):
    max_new, do_sample, temp = _resolve_params(req)

    # fine-tuned
    p_ft = build_prompt(ft_tokenizer, req.text)
    t0 = time.time()
    out_ft = generate_text(ft_model, ft_tokenizer, p_ft, max_new, do_sample, temp)
    t_ft = round(time.time() - t0, 3)

    # base
    p_b = build_prompt(base_tokenizer, req.text)
    t0 = time.time()
    out_b = generate_text(base_model, base_tokenizer, p_b, max_new, do_sample, temp)
    t_b = round(time.time() - t0, 3)

    return {
        "input": req.text,
        "params": {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "temperature": temp,
        },
        "fine_tuned": {
            "model": str(FT_MODEL_DIR),
            "latency_s": t_ft,
            "redacted": out_ft,
        },
        "base": {"model": BASE_MODEL_ID, "latency_s": t_b, "redacted": out_b},
    }
