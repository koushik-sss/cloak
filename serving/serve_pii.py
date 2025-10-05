import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your merged model directory
MODEL_DIR = os.path.expanduser(
    "/Users/saikoushik/Downloads/trainium_stuff/merged_model"
)  # <-- change if needed

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

# Pick best device automatically
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

app = FastAPI(title="PII Redaction Model")


class RedactRequest(BaseModel):
    text: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    do_sample: bool = False


def build_prompt(user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": PII_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # fallback format similar to TinyLlama chat
    return (
        f"<|system|>\n{PII_SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_text}</s>\n"
        f"<|assistant|>\n"
    )


@app.post("/redact")
def redact(req: RedactRequest):
    prompt = build_prompt(req.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = output[0][inputs["input_ids"].shape[-1] :]
    text_out = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return {"redacted": text_out}


@app.get("/")
def root():
    return {"status": "ok", "device": str(device)}
