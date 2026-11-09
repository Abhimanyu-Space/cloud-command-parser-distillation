import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "full2972_v2"
DATA = ROOT / "data" / "validation.jsonl"
OUT = ROOT / "results" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL), local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL), local_files_only=True
)
model.eval()
device = torch.device("cpu")
model.to(device)

def extract_json(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None

def normalize(value):
    if value is None:
        return None
    return str(value).strip().lower()

records = [
    json.loads(line)
    for line in DATA.open(encoding="utf-8")
    if line.strip()
]

predictions = []
intent_correct = 0
slot_correct = {k: 0 for k in ("date", "time", "place")}
exact_correct = 0
parse_success = 0

for index, record in enumerate(records, 1):
    prompt = f"Command: {record['text']}\nJSON: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
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

    pred = extract_json(generated)
    valid = (
        isinstance(pred, dict)
        and all(k in pred for k in ("intent", "date", "time", "place"))
    )

    if valid:
        parse_success += 1
        intent_ok = normalize(pred["intent"]) == normalize(record["intent"])
        intent_correct += int(intent_ok)

        for key in slot_correct:
            slot_correct[key] += int(
                normalize(pred[key]) == normalize(record[key])
            )

        exact_ok = intent_ok and all(
            normalize(pred[k]) == normalize(record[k])
            for k in slot_correct
        )
        exact_correct += int(exact_ok)
    else:
        pred = pred if isinstance(pred, dict) else {"raw_output": generated}
        intent_ok = False
        exact_ok = False

    predictions.append({
        "id": record["id"],
        "text": record["text"],
        "expected": {k: record[k] for k in ("intent", "date", "time", "place")},
        "predicted": pred,
        "parse_success": valid,
        "exact_match": exact_ok
    })

    print(f"Evaluated {index}/{len(records)}", flush=True)

n = len(records)
metrics = {
    "validation_records": n,
    "parse_success": parse_success,
    "parse_success_rate": round(parse_success / n * 100, 2),
    "intent_accuracy": round(intent_correct / n * 100, 2),
    "date_accuracy": round(slot_correct["date"] / n * 100, 2),
    "time_accuracy": round(slot_correct["time"] / n * 100, 2),
    "place_accuracy": round(slot_correct["place"] / n * 100, 2),
    "exact_match_accuracy": round(exact_correct / n * 100, 2)
}

with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for item in predictions:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with (OUT / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n===== VALIDATION RESULTS =====")
print(json.dumps(metrics, indent=2))
print("\nSaved predictions and metrics in:", OUT)
