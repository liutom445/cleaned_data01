import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV        = "trial18.csv"  # Path to your trial18 dataset
OUTPUT_CSV       = "trial18_counterfactuals.csv"
OPENAI_MODEL     = "gpt-4.1"
DEEPSEEK_MODEL   = "deepseek-chat"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build prediction prompt for days until disengagement (90-day gap)."""
    return (
        "Disclaimer: Simulation for educational purposes.\n\n"
        "You are an expert in HIV care retention.\n"
        "Patient baseline data:\n"
        f"• Age category: {row['X_AgeCat_0m']}\n"
        f"• Gender: {row['X_Gender_0m']}\n"
        f"• Years since diagnosis: {row['X_TimeSinceDx_0m']}\n"
        f"• Years in clinic: {row['X_TimeSinceClinic_0m']}\n"
        f"• On ART: {row['X_ARTStatus_0m']}\n"
        f"• # aware of patient's status: {row['X_yourHIVTotal_0m']}\n"
        f"• Material support: {row['X_MaterialSupportMean_0m']}\n"
        f"• Internalized stigma: {row['X_StigInternal_0m']}\n\n"
        f"Assigned intervention: {treatment}.\n"
        "Predict the expected number of days from baseline until the patient has a 90-day\n"
        "gap in care within the first 12 months. Reply with a single numeric value (days)."
    )

# ─── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ─── Initialize LLM clients ───────────────────────────────────────────────────
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# ─── Define counterfactual arms ───────────────────────────────────────────────
arms = {
    "chat_control_days":      "Usual care",
    "chat_intervention_days": "Microclinic intervention"
}

# Prepare Deepseek column keys
ds_arms = {k.replace("chat_", "ds_"): v for k, v in arms.items()}

# Initialize prediction columns
for col in list(arms) + list(ds_arms):
    df[col] = None

# Calculate total expected queries
num_models = 2
total_queries = len(df) * len(arms) * num_models

# ─── Generate predictions with progress & ETA ─────────────────────────────────
start_time = time.time()
query_count = 0

for idx, row in df.iterrows():
    for key, label in arms.items():
        prompt = generate_prompt(row, label)

        # OpenAI
        resp = chat_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        df.at[idx, key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # Deepseek
        ds_key = key.replace("chat_", "ds_")
        resp = ds_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        df.at[idx, ds_key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # Notify progress every 30 queries
        if query_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_per_query = elapsed / query_count
            remaining = total_queries - query_count
            eta = remaining * avg_per_query
            print(f"🛎️ {query_count}/{total_queries} queries done | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

# ─── Save results ─────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")
