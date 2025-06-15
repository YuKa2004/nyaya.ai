
import os
import csv
import json
import requests
from time import sleep

OPENAI_API_KEY = os.getenv("")  # Set your key in env
CSV_FILE = "legal_dataset.csv"  # Name of your CSV file
OUTPUT_JSON = "clauses_embedded.json"
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    response = requests.post(
        url,
        headers=headers,
        json={
            "input": text,
            "model": EMBEDDING_MODEL,
        },
    )
    if not response.ok:
        print(f"Error for text: {text[:50]} - {response.status_code}: {response.text}")
        sleep(2)
        return get_embedding(text)
    return response.json()["data"][0]["embedding"]

data = []
with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        clause = row["clause"]
        id_ = row["id"]
        print(f"Embedding: {clause[:40]}...")
        embedding = get_embedding(clause)
        data.append({"id": id_, "clause": clause, "embedding": embedding})
        sleep(1)  # To respect rate limits

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"Wrote {len(data)} entries to {OUTPUT_JSON}")
