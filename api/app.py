import json
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "full2972_v2"
MODEL_VERSION = "full2972_v2"

app = FastAPI(title="Cloud Command Parser", version="1.0.0")
lock = threading.Lock()

tokenizer = None
model = None

class ParseRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.to(torch.device("cpu"))
    model.eval()

def extract_json(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_VERSION, "loaded": model is not None}

@app.post("/parse")
def parse_command(request: ParseRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    prompt = f"Command: {request.text.strip()}\nJSON: "
    with lock, torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    result = extract_json(generated)
    headers = {
        "X-Served-By": "local-cpu",
        "X-Model-Version": MODEL_VERSION
    }

    if isinstance(result, dict) and all(
        key in result for key in ("intent", "date", "time", "place")
    ):
        return JSONResponse(content=result, headers=headers)

    return JSONResponse(
        content={"error": "Model output was not valid four-field JSON",
                 "raw_output": generated},
        headers=headers
    )
