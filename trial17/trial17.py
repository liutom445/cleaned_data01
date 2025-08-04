import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV        = "trial17.csv"
OUTPUT_CSV       = "trial17_counterfactuals.csv"
OPENAI_MODEL     = "gpt-4.1-mini"            # ChatGPT model
DEEPSEEK_MODEL   = "deepseek-chat"      # Deepseek model

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build prompt for predicting 24-week percent change in LDL-C."""
    sex   = "Male" if row['X_sex_0w'] == 1 else "Female"
    hbp   = "Yes"  if row['X_hbp_0w'] == 1 else "No"
    smoke = "Yes"  if row['X_smoke_0w'] == 1 else "No"
    dm    = "Yes"  if row['X_DM_0w']   == 1 else "No"
    return (
        "Disclaimer: Simulation for educational purposes.\n\n"
        "You are an expert in lipid-lowering therapies.\n"
        "Patient baseline characteristics:\n"
        f"- Age: {row['X_age_0w']} years\n"
        f"- Sex: {sex}\n"
        f"- Hypertension history: {hbp}\n"
        f"- Smoking history: {smoke}\n"
        f"- Diabetes history: {dm}\n"
        f"- Body mass index: {row['X_BMI_0w']:.1f} kg/m²\n"
        f"- Baseline LDL-C: {row['X_LDL_0w']} mg/dL\n"
        f"- Baseline total cholesterol: {row['X_TC_0w']} mg/dL\n"
        f"- Baseline triglycerides: {row['X_TG_0w']} mg/dL\n"
        f"- Baseline HDL-C: {row['X_HDL_0w']} mg/dL\n\n"
        f"Assigned treatment: {treatment}.\n"
        "Predict the expected percent change in LDL-C from baseline to 24 weeks.\n"
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
# ─── Define binary arms (control vs. treatment) ───────────────────────────────
arms = {
    "chat_control_delta":   "rosuvastatin 20 mg daily",
    "chat_treatment_delta": "rosuvastatin 10 mg daily + alirocumab 75 mg q2w"
}
ds_arms = {k.replace("chat_", "ds_"): v for k, v in arms.items()}

# ─── Prepare output columns ────────────────────────────────────────────────────
for col in list(arms) + list(ds_arms):
    df[col] = None

total_queries = len(df) * len(arms) * 2
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

        # Progress every 30 calls
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
