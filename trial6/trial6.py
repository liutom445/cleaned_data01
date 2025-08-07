import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV        = "trial6/trial6.csv"                   # Path to your trial6 dataset
OUTPUT_CSV       = "trial6/trial6_counterfactuals.csv"   # Where to save predictions
OPENAI_MODEL     = "gpt-4.1-mini"                  # LLM for counterfactuals
DEEPSEEK_MODEL   = "deepseek-chat"                # Deepseek model

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build prompt for predicting 24‑month CCA IMT change."""
    sex = "Male" if row['X_sex_0m'] == 1 else "Female"
    statin = "Yes" if row['X_statin_0m'] == 1 else "No"
    return (
        "Disclaimer: Simulation for educational purposes.\n\n"
        "You are an expert in diabetes and cardiovascular research.\n"
        "Patient baseline characteristics:\n"
        f"- Age: {row['X_age_0m']} years\n"
        f"- Sex: {sex}\n"
        f"- Statin use: {statin}\n"
        f"- Pre-randomization diabetes therapy: {row['X_dm_treatment_0m']}\n"
        f"- Baseline systolic BP: {row['X_sbp_0m']} mmHg\n"
        f"- Baseline HbA1c: {row['X_hba1c_ancova_0m']} %\n"
        f"- Baseline mean CCA IMT: {row['X_cca_imt_0m']} mm\n"
        f"- Baseline maximum IMT: {row['X_max_imt_0m']} mm\n\n"
        f"Assigned treatment: {treatment}.\n"
        "Predict the expected 24‑month change in mean common carotid artery IMT (in mm).\n"
        "Reply with a single numeric value."
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
    "chat_sitagliptin_delta":      "Sitagliptin",
    "chat_conventional_delta":     "Conventional"
}
ds_arms = {k.replace("chat_", "ds_"): v for k, v in arms.items()}

# Initialize columns for predictions
for col in list(arms) + list(ds_arms):
    df[col] = None

# Total queries for progress tracking
total_queries = len(df) * len(arms) * 2
start_time    = time.time()
query_count   = 0

# ─── Generate predictions ─────────────────────────────────────────────────────
for idx, row in df.iterrows():
    for key, label in arms.items():
        prompt = generate_prompt(row, label)

        # GPT‑O3‑Mini prediction
        resp = chat_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        df.at[idx, key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # Deepseek prediction
        ds_key = key.replace("chat_", "ds_")
        resp   = ds_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        df.at[idx, ds_key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # Progress update every 30 queries
        if query_count % 30 == 0:
            elapsed   = time.time() - start_time
            avg_time  = elapsed / query_count
            remaining = total_queries - query_count
            eta       = remaining * avg_time
            print(f"🛎️ {query_count}/{total_queries} queries | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

# ─── Save results ─────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")
