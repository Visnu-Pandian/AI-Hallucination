import json
from openai import OpenAI
import time

client = OpenAI(api_key="")

with open("hallucinations.json", encoding="utf-8") as f:
    hallucinations = [json.loads(line) for line in f if line.strip()]

CHUNK_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.5

chunks = [hallucinations[i:i+CHUNK_SIZE] for i in range(0, len(hallucinations), CHUNK_SIZE)]

final_ordered_ratings = []


def safe_json_load(output):
    try:
        return json.loads(output)
    except Exception:
        cleaned = output.strip().strip("`").strip()
        try:
            return json.loads(cleaned)
        except:
            return None


def validate_output(parsed, chunk_size):
    """
    Validate:
    - Has 'ratings'
    - Is list
    - Correct number of items
    - Each item has severity:int and justification:string
    """
    if not parsed or "ratings" not in parsed:
        return False

    ratings = parsed["ratings"]

    if not isinstance(ratings, list):
        return False

    # Must match number of hallucinations
    if len(ratings) != chunk_size:
        return False

    # Validate each rating object
    for r in ratings:
        if not isinstance(r, dict):
            return False

        if "severity" not in r or "justification" not in r:
            return False

        if not isinstance(r["severity"], int):
            return False

        if not isinstance(r["justification"], str):
            return False

    return True


for index, chunk in enumerate(chunks, start=1):

    print(f"\n▶ Processing chunk {index}/{len(chunks)} ({len(chunk)} hallucinations)")

    # Structured prompt
    prompt_payload = {
        "task": "Rate hallucination severity.",
        "instructions": {
            "rules": [
                "You MUST return exactly one rating per hallucination.",
                "Maintain the same order as the input.",
                "Each rating MUST contain only: severity (integer 1-10) and justification (string).",
                "Do NOT include ids, indexes, or extra fields.",
                "Do NOT skip, merge, or summarize hallucinations."
            ],
            "severity_scale": {
                "1-3": "Minor",
                "4-6": "Moderate",
                "7-8": "Severe",
                "9-10": "Critical"
            }
        },
        "output_format": {
            "ratings": [
                {
                    "severity": "int",
                    "justification": "string"
                }
            ]
        },
        "input_hallucinations": chunk
    }

    parsed = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"   🔄 Attempt {attempt}/{MAX_RETRIES}")

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output ONLY valid JSON following the schema: "
                        "{ 'ratings': [ {'severity': int, 'justification': string} ] }. "
                        "NEVER include ids, indexes, or extra fields."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False)
                }
            ],
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            verbosity="low"
        )

        raw_output = response.choices[0].message.content
        parsed = safe_json_load(raw_output)

        if validate_output(parsed, len(chunk)):
            print("   ✅ Valid JSON and correct rating count + types.")
            break

        print("   ❌ Invalid output — retrying chunk...")
        time.sleep(RETRY_DELAY)
        parsed = None

    if parsed is None:
        print("   ❌ Chunk FAILED after retries — inserting fallback ratings.")

        parsed = {
            "ratings": [
                {
                    "severity": -1,
                    "justification": (
                        f"Fallback rating: chunk {index}, item {i+1} "
                        f"failed strict JSON/type/count validation."
                    )
                }
                for i in range(len(chunk))
            ]
        }

    # Append validated or fallback ratings
    final_ordered_ratings.extend(parsed["ratings"])


with open("hallucination_ratings.json", "w", encoding="utf-8") as f:
    json.dump({"ratings": final_ordered_ratings}, f, indent=2, ensure_ascii=False)

print("\n🎉 DONE — Strict validation complete and ratings saved to hallucination_ratings.json")
