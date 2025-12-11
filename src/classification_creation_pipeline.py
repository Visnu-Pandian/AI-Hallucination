import json
from openai import OpenAI
import time

client = OpenAI(api_key="")

with open("hallucinations.json", encoding="utf-8") as f:
    hallucinations = [json.loads(line) for line in f if line.strip()]

CHUNK_SIZE = 20 
MAX_RETRIES = 3
RETRY_DELAY = 1.2

chunks = [hallucinations[i:i+CHUNK_SIZE] for i in range(0, len(hallucinations), CHUNK_SIZE)]
all_classifications = []   # raw classifications before summarization

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
    Expected schema:
    {
        "classifications": [
            "topic string",
            "topic string",
            ...
        ]
    }
    """
    if not parsed or "classifications" not in parsed:
        return False

    cl = parsed["classifications"]

    if not isinstance(cl, list):
        return False
    if len(cl) != chunk_size:
        return False

    # Every classification must be a string
    if not all(isinstance(x, str) for x in cl):
        return False

    return True

for index, chunk in enumerate(chunks, start=1):

    print(f"\n▶ Processing chunk {index}/{len(chunks)} ({len(chunk)} hallucinations)")

    prompt_payload = {
        "task": "Classify each hallucination into a short topic label.",
        "instructions": {
            "rules": [
                "You MUST return exactly one topic classification per hallucination.",
                "Output MUST be a JSON object with a 'classifications' array.",
                "Maintain input order.",
                "Each classification MUST be a SHORT string topic, e.g., 'math', 'medicine', 'legal', 'coding', 'history', etc."
            ]
        },
        "input_hallucinations": chunk,
        "output_format": {
            "classifications": ["string"]
        }
    }

    parsed = None

    # Retry loop
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"   🔄 Attempt {attempt}/{MAX_RETRIES}...")

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON in the format: "
                        "{ 'classifications': ['topic', 'topic', ...] }. "
                        "Exactly one topic string per hallucination."
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
            print("   ✅ Valid JSON and correct classification count.")
            break

        print("   ❌ Invalid output — retrying...")
        parsed = None
        time.sleep(RETRY_DELAY)

    # Fallback if still invalid
    if parsed is None:
        print("   ❌ Chunk FAILED after retries — inserting fallback topics.")
        parsed = {
            "classifications": [
                f"unknown_topic_chunk_{index}_item_{i+1}"
                for i in range(len(chunk))
            ]
        }

    # Add to overall list
    all_classifications.extend(parsed["classifications"])

print("\n📘 Summarizing classifications...")

summary_prompt = {
    "task": "Summarize classification topics.",
    "instructions": [
        "Given a long list of classification labels, merge similar ones.",
        "Return a SHORT list of 25–30 high-level topic categories.",
        "Do NOT include duplicates.",
        "Use simple category names (e.g., 'math', 'science', 'legal', 'history')."
    ],
    "input_classifications": all_classifications,
    "output_format": {
        "summary_topics": ["string"]
    }
}

summary_response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "Return ONLY valid JSON in the format: "
                "{ 'summary_topics': ['topic', 'topic', ...] }. "
                "Produce ~25–30 categories."
            )
        },
        {
            "role": "user",
            "content": json.dumps(summary_prompt, ensure_ascii=False)
        }
    ],
    response_format={"type": "json_object"},
    reasoning_effort="minimal",
    verbosity="low"
)

summary_parsed = safe_json_load(summary_response.choices[0].message.content)

with open("topics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_parsed, f, indent=2, ensure_ascii=False)

print("\nDONE — Summary saved to topics_summary.json")
