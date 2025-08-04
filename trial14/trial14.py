import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV      = "trial14.csv"
OUTPUT_CSV     = "trial14_counterfactuals.csv"
OPENAI_MODEL   = "o4-mini"           # ChatGPT
DEEPSEEK_MODEL = "deepseek-chat"     # Deepseek

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def bool_str(val: int) -> str:
    return "Yes" if val == 1 else "No"

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build prompt for predicting 6-week change in 6MWD."""
    return (
        "Disclaimer: Simulation for educational purposes only.\n\n"
        "You are an expert in pulmonary rehabilitation.\n"
        "Patient baseline characteristics:\n"
        f"- Diabetes: {bool_str(row['X_diabetes_0w'])}\n"
        f"- Heart disease: {bool_str(row['X_heart_0w'])}\n"
        f"- Chronic lung disease: {bool_str(row['X_lung_0w'])}\n"
        f"- Obesity: {bool_str(row['X_obesity_0w'])}\n"
        f"- Hypertension: {bool_str(row['X_hypertension_0w'])}\n"
        f"- Other comorbidities: {bool_str(row['X_othercomorb_0w'])}\n"
        f"- Smoking history: {bool_str(row['X_smoke_0w'])}\n"
        f"- Baseline 6MWD: {row['X_6MWD_0w']} meters\n"
        f"- Baseline squat time: {row['X_squat_0w']} seconds\n"
        f"- Baseline FVC: {row['X_fvc_0w']:.2f} L\n"
        f"- Baseline FEV1: {row['X_fev1_0w']:.2f} L\n"
        f"- Baseline FEV1/FVC: {row['X_fevfvc_0w']:.2f}\n"
        f"- Baseline MVV: {row['X_mvv_0w']:.2f} L/min\n"
        f"- Baseline PEF: {row['X_pef_0w']:.2f} L/s\n"
        f"- SF-12 PCS: {row['X_SF12_PCS_0w']:.1f}\n"
        f"- SF-12 MCS: {row['X_SF12_MCS_0w']:.1f}\n\n"
        f"Assigned intervention: {treatment}.\n"
        "Predict the expected change in 6-minute walking distance (in meters) from baseline to 6 weeks.\n"
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

# ─── Define binary arms ────────────────────────────────────────────────────────
arms = {
    "chat_control_delta":   "usual care without structured exercise programme",
    "chat_treatment_delta": "6-week home-based TERECO exercise programme"
}
ds_arms = {k.replace("chat_", "ds_"): v for k, v in arms.items()}

# ─── Prepare output columns ────────────────────────────────────────────────────
for col in list(arms) + list(ds_arms):
    df[col] = None

total_queries = len(df) * (len(arms) * 2)
start_time    = time.time()
query_count   = 0

# ─── Generate counterfactual predictions ───────────────────────────────────────
for idx, row in df.iterrows():
    for key, label in arms.items():
        prompt = generate_prompt(row, label)

        # — ChatGPT prediction
        resp = chat_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        df.at[idx, key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # — Deepseek prediction
        ds_key = key.replace("chat_", "ds_")
        resp   = ds_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        df.at[idx, ds_key] = parse_numeric(resp.choices[0].message.content)
        query_count += 1

        # Print progress every 30 calls
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
