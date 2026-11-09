
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


# --------------------------------------------------
# PATHS AND CONSTANTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "training" / "config.yaml"

ALLOWED_INTENTS = {
    "alarm_set",
    "alarm_query",
    "calendar_set",
    "calendar_query",
    "calendar_remove",
    "weather_query",
    "datetime_query",
    "recommendation_events",
}

TARGET_KEYS = ["intent", "date", "time", "place"]


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# CONFIGURATION AND DATA
# --------------------------------------------------

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid training configuration.")

    return config


def load_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    records = []
    seen_ids = set()

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

            required = {"id", "text", *TARGET_KEYS}

            if not required.issubset(record):
                raise ValueError(
                    f"Missing fields at line {line_number}."
                )

            record_id = record["id"]

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate training ID: {record_id}"
                )

            seen_ids.add(record_id)

            if not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError(
                    f"Invalid command text at line {line_number}."
                )

            if record["intent"] not in ALLOWED_INTENTS:
                raise ValueError(
                    f"Invalid intent in record {record_id}: "
                    f"{record['intent']}"
                )

            for key in ("date", "time", "place"):
                value = record[key]

                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"Invalid {key} in record {record_id}."
                    )

                if isinstance(value, str) and value.strip().lower() == "null":
                    raise ValueError(
                        f"String 'null' found in {key} of {record_id}. "
                        "Use actual JSON null."
                    )

            records.append(record)

    return records


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class CommandDataset(Dataset):

    def __init__(self, records, tokenizer, max_length):
        self.examples = []

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token.")

        for record in records:

            prompt = (
                f"Command: {record['text']}\n"
                "JSON: "
            )

            target_obj = {
                "intent": record["intent"],
                "date": record["date"],
                "time": record["time"],
                "place": record["place"],
            }

            target = json.dumps(
                target_obj,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            )["input_ids"]

            eos_id = tokenizer.eos_token_id

            # The complete target and EOS must fit.
            if len(target_ids) + 1 > max_length:
                raise ValueError(
                    f"Target too long for record {record['id']}."
                )

            available_prompt = (
                max_length - len(target_ids) - 1
            )

            # Keep the end of the prompt if truncation is needed.
            if available_prompt > 0:
                prompt_ids = prompt_ids[-available_prompt:]
            else:
                prompt_ids = []

            input_ids = (
                prompt_ids + target_ids + [eos_id]
            )

            # Train only on the JSON response and EOS.
            labels = (
                [-100] * len(prompt_ids)
                + target_ids
                + [eos_id]
            )

            if not any(x != -100 for x in labels[1:]):
                raise ValueError(
                    f"No valid target tokens for {record['id']}."
                )

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        if not self.examples:
            raise ValueError("Dataset contains no examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# --------------------------------------------------
# BATCH PADDING
# --------------------------------------------------

def collate_batch(batch, pad_id):

    max_len = max(
        len(item["input_ids"]) for item in batch
    )

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:

        padding = max_len - len(item["input_ids"])

        input_ids.append(
            item["input_ids"] + [pad_id] * padding
        )

        attention_masks.append(
            [1] * len(item["input_ids"])
            + [0] * padding
        )

        labels.append(
            item["labels"] + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(
            input_ids, dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(
            labels, dtype=torch.long
        ),
    }


# --------------------------------------------------
# CHECKPOINT SAVING
# --------------------------------------------------

def save_checkpoint(model, tokenizer, path):

    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(path)

    print(f"Checkpoint saved: {path}", flush=True)


# --------------------------------------------------
# FINITE VALUE CHECKS
# --------------------------------------------------

def check_model_parameters(model):

    for name, parameter in model.named_parameters():

        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"Non-finite model parameter detected: {name}"
            )


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train the command parser student model."
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config = load_config()

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.subset is not None and args.subset < 1:
        raise ValueError("Subset must be at least 1.")

    if args.run_name is not None:
        if (
            not args.run_name
            or Path(args.run_name).name != args.run_name
            or args.run_name in {".", ".."}
        ):
            raise ValueError("Invalid run name.")

    # --------------------------------------------------
    # SEED AND DEVICE
    # --------------------------------------------------

    seed = int(config["project"]["seed"])
    set_seed(seed)

    # This assignment uses CPU training.
    device = torch.device("cpu")

    print("Device:", device)

    # --------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------

    model_name = config["model"]["name"]
    max_length = int(config["model"]["max_length"])

    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    accumulation = int(
        config["training"]["gradient_accumulation_steps"]
    )

    learning_rate = float(
        config["training"]["learning_rate"]
    )

    weight_decay = float(
        config["training"]["weight_decay"]
    )

    warmup_ratio = float(
        config["training"]["warmup_ratio"]
    )

    if epochs < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError(
            "Epochs, batch size and accumulation must be positive."
        )

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    if not 0 <= warmup_ratio < 1:
        raise ValueError("Warmup ratio must be in [0, 1).")

    train_path = ROOT / config["data"]["train_file"]

    output_dir = ROOT / config["output"]["results_directory"]

    v1_path = ROOT / config["output"]["checkpoint_v1"]
    v2_path = ROOT / config["output"]["checkpoint_v2"]

    if args.run_name:
        output_dir = output_dir / args.run_name

        v1_path = (
            ROOT / "models" / f"{args.run_name}_v1"
        )

        v2_path = (
            ROOT / "models" / f"{args.run_name}_v2"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # LOAD TRAINING DATA ONLY
    # --------------------------------------------------

    records = load_records(train_path)

    if args.subset is not None:
        records = records[:args.subset]

    if not records:
        raise ValueError("Training dataset is empty.")

    print("Training records:", len(records))
    print("Validation data is NOT loaded or used.")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Gradient accumulation:", accumulation)
    print("Learning rate:", learning_rate)

    # --------------------------------------------------
    # LOAD CACHED TOKENIZER AND MODEL
    # --------------------------------------------------

    print("\nLoading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS token is missing.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading student model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.to(
        device=device,
        dtype=torch.float32,
    )

    check_model_parameters(model)

    # --------------------------------------------------
    # CREATE DATASET AND DATALOADER
    # --------------------------------------------------

    dataset = CommandDataset(
        records,
        tokenizer,
        max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(
            batch,
            tokenizer.pad_token_id,
        ),
    )

    print("Training batches per epoch:", len(loader))

    # --------------------------------------------------
    # OPTIMIZER AND SCHEDULER
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    updates_per_epoch = math.ceil(
        len(loader) / accumulation
    )

    total_steps = updates_per_epoch * epochs

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --------------------------------------------------
    # LOSS CSV
    # --------------------------------------------------

    loss_file = output_dir / "loss.csv"

    with open(
        loss_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "learning_rate",
            "optimizer_steps",
            "target_tokens",
        ])

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_target_tokens = 0
        optimizer_steps = 0

        print(
            f"\nStarting epoch {epoch + 1}/{epochs}",
            flush=True,
        )

        for step, batch in enumerate(loader):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # Forward pass.
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )

            # Explicit causal language-model loss.
            shift_logits = outputs.logits[:, :-1, :].contiguous()

            shift_labels = batch["labels"][:, 1:].contiguous()

            valid_tokens = int(
                (shift_labels != -100).sum().item()
            )

            if valid_tokens == 0:
                raise RuntimeError(
                    f"No target tokens in batch {step + 1}."
                )

            if not torch.isfinite(shift_logits).all():
                raise RuntimeError(
                    f"Non-finite logits at batch {step + 1}."
                )

            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1),
                ),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {step + 1}. "
                    f"Valid target tokens: {valid_tokens}."
                )

            batch_loss = loss.item()

            total_loss += batch_loss * valid_tokens
            total_target_tokens += valid_tokens

            # Correctly scale incomplete accumulation groups.
            group_start = (
                step // accumulation
            ) * accumulation

            group_size = min(
                accumulation,
                len(loader) - group_start,
            )

            scaled_loss = loss / group_size
            scaled_loss.backward()

            # Optimizer update.
            is_update_step = (
                (step + 1) % accumulation == 0
                or (step + 1) == len(loader)
            )

            if is_update_step:

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        "Non-finite gradient norm detected."
                    )

                optimizer.step()

                # Detect numerical corruption immediately.
                check_model_parameters(model)

                scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                optimizer_steps += 1

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {step + 1}/{len(loader)} | "
                f"Loss: {batch_loss:.6f} | "
                f"Target tokens: {valid_tokens}",
                flush=True,
            )

        # --------------------------------------------------
        # EPOCH RESULTS
        # --------------------------------------------------

        if total_target_tokens == 0:
            raise RuntimeError(
                "No target tokens were processed."
            )

        average_loss = (
            total_loss / total_target_tokens
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if not math.isfinite(average_loss):
            raise RuntimeError(
                "Epoch loss is not finite."
            )

        print(
            f"\nEpoch {epoch + 1}/{epochs} completed"
        )

        print(
            f"Average token loss: {average_loss:.6f}"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        print(
            f"Optimizer updates: {optimizer_steps}"
        )

        # Write completed epoch results.
        with open(
            loss_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                average_loss,
                current_lr,
                optimizer_steps,
                total_target_tokens,
            ])

        # Save v1 after first epoch.
        if epoch == 0:
            save_checkpoint(
                model,
                tokenizer,
                v1_path,
            )

    # --------------------------------------------------
    # SAVE FINAL CHECKPOINT
    # --------------------------------------------------

    save_checkpoint(
        model,
        tokenizer,
        v2_path,
    )

    print("\nTraining completed successfully.")

    print("v1 checkpoint:", v1_path)
    print("v2 checkpoint:", v2_path)
    print("Loss history:", loss_file)


if __name__ == "__main__":
    main()
