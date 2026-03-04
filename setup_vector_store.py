"""
setup_vector_store.py
=====================
Run ONCE before starting the Life Coach app.
Creates a Vector Store, uploads the goals file, saves ID to .env.

Usage:
    python setup_vector_store.py
"""

import os, time
from openai import OpenAI
from pathlib import Path

client = OpenAI()

GOALS_FILE = "my_goals.txt"
STORE_NAME = "Life Coach Goals"
ENV_FILE = ".env"

# 1. Create vector store
print("[1/4] Creating vector store...")
store = client.vector_stores.create(name=STORE_NAME)
print(f"  → ID: {store.id}")

# 2. Upload file
print(f"[2/4] Uploading '{GOALS_FILE}'...")
with open(GOALS_FILE, "rb") as f:
    uploaded = client.files.create(file=f, purpose="assistants")
print(f"  → File ID: {uploaded.id}")

# 3. Index file in vector store
print("[3/4] Indexing file...")
client.vector_stores.files.create(
    vector_store_id=store.id,
    file_id=uploaded.id,
)
while True:
    status = client.vector_stores.files.retrieve(
        vector_store_id=store.id, file_id=uploaded.id
    ).status
    print(f"  → Status: {status}")
    if status == "completed":
        break
    if status == "failed":
        print("  ERROR: indexing failed")
        exit(1)
    time.sleep(1)

# 4. Save to .env
print("[4/4] Saving to .env...")
env = Path(ENV_FILE)
lines = []
if env.exists():
    lines = [l for l in env.read_text().splitlines(True) if not l.startswith("VECTOR_STORE_ID=")]
with open(ENV_FILE, "w") as f:
    f.writelines(lines)
    f.write(f"VECTOR_STORE_ID={store.id}\n")

print(f"\n✅ Done! VECTOR_STORE_ID={store.id}")
print("Run: streamlit run life_coach_final.py")