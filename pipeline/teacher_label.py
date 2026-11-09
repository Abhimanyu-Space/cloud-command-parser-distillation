import argparse
import json
import logging
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "commands_pool.jsonl"
SCHEMA_FILE = ROOT / "data" / "schema.json"
OUTPUT_DIR = ROOT / "results" / "teacher"


MODEL = "phi3.5:3.8b"
API = "http://localhost:11434/api/generate"
KEYS = {"intent", "date", "time", "place"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WRITE_LOCK = threading.Lock()

RAW_FILE = OUTPUT_DIR / "teacher_raw.jsonl"
VALID_FILE = OUTPUT_DIR / "teacher_labeled.jsonl"
FAILED_FILE = OUTPUT_DIR / "teacher_failed.jsonl"
LOG_FILE = ROOT / "results" / "logs" / "teacher.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path, record):
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def completed_ids():
    if not VALID_FILE.exists():
        return set()
    return {r["id"] for r in load_jsonl(VALID_FILE)}


def validate(raw, command, allowed_intents):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json", []

    if not isinstance(obj, dict) or set(obj) != KEYS:
        return None, "incorrect_keys", []

    if not isinstance(obj["intent"], str):
        return None, "invalid_intent_type", []

    if obj["intent"] not in allowed_intents:
        return None, "invalid_intent", []

    repairs = []

    for slot in ("date", "time", "place"):
        value = obj[slot]

        # Repair the literal string "null" to actual JSON null.
        if isinstance(value, str) and value.strip().lower() == "null":
            obj[slot] = None
            repairs.append(f"{slot}: string_null_to_json_null")
            continue

        if value is None:
            continue

        if not isinstance(value, str) or not value.strip():
            return None, f"invalid_{slot}_type_or_empty", repairs

        # Exact match: no repair needed.
        if value in command:
            continue

        # Case-only correction, but only when the match is unique.
        matches = list(
            re.finditer(re.escape(value), command, flags=re.IGNORECASE)
        )

        if len(matches) == 1:
            match = matches[0]
            obj[slot] = command[match.start():match.end()]
            repairs.append(f"{slot}: capitalization_corrected")
        else:
            return None, f"{slot}_not_exact_or_ambiguous", repairs

    return obj, None, repairs

def label(command, json_schema):

    system = (
        "You are a strict English command parser. "
        "Classify the user's command into exactly one of these intents: "
        "alarm_set: create an alarm; "
        "alarm_query: ask about existing alarms; "
        "calendar_set: add an event or create a reminder; "
        "calendar_query: ask about scheduled events or plans; "
        "calendar_remove: delete an existing event; "
        "weather_query: ask about weather; "
        "datetime_query: ask for the current date or time; "
        "recommendation_events: ask for events to attend. "

        "Distinguish carefully between creating, querying and deleting. "
        "A request to remind someone or create a reminder is calendar_set, "
        "not calendar_remove or datetime_query. "
        "A question asking whether an appointment or schedule exists "
        "is calendar_query, not calendar_set. "
        "A question asking what time it is in a city is datetime_query, "
        "not weather_query. "
        "A question about weather conditions is weather_query. "

        "A request to wake someone up, set an alarm or arrange an alarm "
        "is alarm_set, even if it uses words such as wake me up. "
        "Do not classify an alarm creation request as calendar_set. "
        "A question about an existing meeting or appointment is calendar_query, "
        "even when it contains the phrase what time. "

        "For time, extract clock times and durations such as "
        "two hours or in two hours, following the shortest relevant "
        "phrase rule. "
        "Never classify an activity, event, person or object as a place. "
        "A place must be a geographical location explicitly mentioned. "

        "Preserve the exact original spelling and capitalization "
        "of every extracted slot. Never capitalize, normalize, abbreviate, "
        "reformat or paraphrase extracted values. "

        "Never convert written-out dates into numeric or abbreviated dates. "
        "For example, twenty one january two thousand and seventeen "
        "must remain exactly that phrase. "

        "Never infer or invent dates such as today or tomorrow. "
        "Only extract dates explicitly present in the command. "

        "Clock times and time-of-day expressions belong in the time slot, "
        "not the date slot. Examples include five pm, three am and lunchtime. "
        "Only extract a time when the command contains an actual clock time, "
        "duration, or time-of-day expression. Words such as repeat, schedule, "
        "meeting and reminder are not time values. "
        "Never assume or invent a time such as now when no time is "
        "explicitly mentioned. An alarm request without a specified time "
        "must have time set to JSON null. "
        "Time expressions must be copied as one exact, continuous substring "
"from the original command. Never insert, remove, repeat, or change "
"words or units when extracting a time range. "
"For example, in 'from five to six hundred am', extract "
"'five to six hundred am', not 'five am to six hundred am'. "

        "When a command contains a time range, extract the complete range "
        "including both endpoints and the connecting words, exactly as "
        "written in the command. Do not extract only one endpoint. "
        "Do not convert or normalize the range. "

        "Expressions such as lunchtime, breakfast time and midnight belong "
        "in the time slot, never the date slot. "

        "If a question asks what time an event is scheduled but does not "
        "provide an actual time expression, return time as null. "
        "Never describe or paraphrase the missing time. "

        "Copy geographical locations as exact contiguous text from the command. "
        "Do not change capitalization, insert commas, or add punctuation. "

        "Words such as 'what time' are question phrases, not time values. "
        "Do not extract question wording as a time. "

        "Geographical names must be copied exactly as they appear. "
        "Do not add commas, punctuation or alternative spellings. "

        "A request to create a reminder, including a reminder for lunchtime, "
        "is calendar_set. "

        "Extract explicit date and weekday ranges completely, including "
        "both endpoints and connecting words. For example, "
        "'monday through friday' must be extracted exactly as "
        "'monday through friday'. Never omit an explicitly stated date range. "
        "Extract date, time and place only when explicitly mentioned. "
        "Date means a day, date or recurring day expression. "
        "Time means a clock time or time expression. "
        "Place means a geographical location, city or region only. "
        "Never classify a person, activity, object or action as a place. "

        "Copy each slot as the shortest relevant phrase exactly as it "
        "appears in the command. Do not normalize or paraphrase. "
        "Exclude prepositions such as at, on, in and for. "
        "For absent slots, output JSON null without quotation marks. "
        "Never output the string 'null'. Never invent information. "

        "Example: set an alarm for seven thirty am tomorrow "
        "produces intent alarm_set, date tomorrow, "
        "time seven thirty am, place null. "

        "Examples of correct slot extraction: "

        "Command: set an event happy birthday john to repeat on "
        "twenty one january two thousand and seventeen. "
        "Output: {\"intent\":\"calendar_set\","
        "\"date\":\"twenty one january two thousand and seventeen\","
        "\"time\":null,\"place\":null}. "

        "Command: what is the current time in lagos nigeria. "
        "Output: {\"intent\":\"datetime_query\","
        "\"date\":null,\"time\":null,\"place\":\"lagos nigeria\"}. "

        "Command: set a reminder alarm for my five pm meeting please. "
        "Output: {\"intent\":\"calendar_set\","
        "\"date\":null,\"time\":\"five pm\",\"place\":null}. "

        "Command: please arrange to wake me up at three am alarm. "
        "Output: {\"intent\":\"alarm_set\","
        "\"date\":null,\"time\":\"three am\",\"place\":null}. "

        "Command: what time am i meeting john tomorrow. "
        "Output: {\"intent\":\"calendar_query\","
        "\"date\":\"tomorrow\",\"time\":null,\"place\":null}. "

        "Command: remind me when it is lunchtime. "
        "Output: {\"intent\":\"calendar_set\","
        "\"date\":null,\"time\":\"lunchtime\",\"place\":null}. "

                "CRITICAL EXACT EXTRACTION RULES: "

        "Never correct spelling, grammar or wording in extracted slots. "
        "Copy the original command exactly, including misspellings. "
        "For example, if the command says 'twelveth', never change it "
        "to 'twelfth'. "

        "Never infer a date from vague wording. "
        "For example, 'for the day' does not mean 'today'. "
        "Only extract a date phrase that is explicitly present. "

        "When multiple times appear, do not combine separate time "
        "phrases into one value. Select the time most directly "
        "associated with the requested action. "
        "For example, in 'remind me at two pm tuesday for the "
        "three pm meeting', extract 'two pm' as the reminder time. "

        "For a time query, do not extract the subject of the query "
        "as the answer. For example, in 'what time does the sun set', "
        "time must be JSON null because no actual time is supplied. "

        "Never convert shorthand geographical names into their "
        "expanded forms. If the command says 'boston', output "
        "'boston', not 'Boston, Massachusetts'. "
        "If the command says 'ny', output 'ny', not 'new york'. "

        "A person's name, event, activity, object or vague description "
        "of a location is not a geographical place. "
        "For example, 'my sister's house' must not be extracted "
        "as a place unless an explicit geographical name is given. "

        "Every non-null slot must be an exact continuous substring "
        "of the original command. Never join separate phrases, "
        "insert words, remove words or paraphrase. "
        "Return only one valid JSON object with exactly these four keys: "
        "intent, date, time, place."
)


    payload = {
        "model": MODEL,
        "system": system,
        "prompt": "Parse this command:\n" + command,
        "format": json_schema,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }

    response = requests.post(API, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise ValueError("Ollama response has no response field")

    return data["response"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
    "--retry-failed",
    action="store_true",
    help="Retry failed IDs that do not yet have accepted labels"
)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    schema_data = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    json_schema = schema_data["json_schema"]
    allowed_intents = set(schema_data["intents"])

    commands = load_jsonl(INPUT)
    if args.limit > 0:
        commands = commands[:args.limit]

    done = completed_ids()

    if args.retry_failed:
        failed_records = load_jsonl(FAILED_FILE) if FAILED_FILE.exists() else []
        failed_ids = {r["id"] for r in failed_records}

        pending = [
            r for r in commands
            if r["id"] in failed_ids and r["id"] not in done
    ]
    else:
        pending = [r for r in commands if r["id"] not in done]

    print(f"Model: {MODEL}")
    print(f"Commands selected: {len(commands)}")
    print(f"Already completed: {len(done)}")
    print(f"Remaining: {len(pending)}")

    if args.workers < 1:
            parser.error("--workers must be at least 1")

    def process_record(index, record):
        command_id = record["id"]
        command = record["text"]

        try:
            time.sleep(max(0, args.delay))
            raw = label(command, json_schema)

            # Preserve the exact teacher response.
            append_jsonl(RAW_FILE, {
                "id": command_id,
                "prediction": raw
            })

            obj, error, repairs = validate(
                raw, command, allowed_intents
            )

            if repairs:
                append_jsonl(
                    OUTPUT_DIR / "teacher_repairs.jsonl",
                    {"id": command_id, "repairs": repairs}
                )

            if error:
                append_jsonl(FAILED_FILE, {
                    "id": command_id,
                    "text": command,
                    "raw": raw,
                    "error": error
                })
                logging.warning(
                    "%s failed validation: %s", command_id, error
                )
                return index, command_id, "INVALID"

            append_jsonl(VALID_FILE, {
                "id": command_id,
                "text": command,
                **obj
            })
            logging.info("%s labelled successfully", command_id)
            return index, command_id, "OK"

        except Exception as exc:
            append_jsonl(FAILED_FILE, {
                "id": command_id,
                "text": command,
                "error": str(exc)
            })
            logging.exception("%s request failed", command_id)
            return index, command_id, f"ERROR - {exc}"

    print(f"Starting parallel labelling with {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_record, i, record)
            for i, record in enumerate(pending, 1)
        ]

        for future in as_completed(futures):
            index, command_id, status = future.result()
            print(f"[{index}/{len(pending)}] {command_id}: {status}")
    print("\nLabelling run finished.")
    print(f"Raw responses: {RAW_FILE}")
    print(f"Valid labels: {VALID_FILE}")
    print(f"Failures: {FAILED_FILE}")

if __name__ == "__main__":
    main()
