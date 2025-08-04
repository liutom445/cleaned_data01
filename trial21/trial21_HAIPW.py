
#!/usr/bin/env python3
"""
trial21_predict_days_simple.py

Generate counterfactual days-to-TB-initiation predictions for the Trial 21 dataset
using two LLMs (OpenAI GPT and Deepseek), without argparse.

Configure input/output filenames and model names directly in the script.
"""

import os
import re
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV        = "trial21.csv"
OUTPUT_CSV       = "trial21_days_predictions.csv"
OPENAI_MODEL     = "gpt-4.1"
DEEPSEEK_MODEL   = "deepseek-chat"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    match = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(match.group(0))

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build the prediction prompt for days to TB treatment initiation."""
    return (
        "Disclaimer: This is an educational simulation only.\n\n"
        "You are an expert in TB diagnosis pathways.\n"
        "Patient baseline data:\n"
        f"• Duration of cough: {row['X_duration_of_cough_weeks_0d']} weeks\n"
        f"• Night sweats: {row['X_night_sweats_0d']}\n"
        f"• Weight loss: {row['X_weight_loss_0d']}\n"
        f"• Fever: {row['X_fever_0d']}\n"
        f"• Reported HIV status: {row['X_reported_hiv_status_0d']}\n"
        f"• On ART: {row['X_antiretroviral_therapy_0d']}\n"
        f"• Baseline EQ5D score: {row['X_eq5d_0d']}\n\n"
        f"Assigned intervention: {treatment}.\n"
        "Predict the expected number of calendar days from baseline to start of TB treatment.\n"
        "Provide a single numeric value (days)."
    )

def get_predictions(client: OpenAI, model_name: str, df: pd.DataFrame, arms: dict) -> pd.DataFrame:
    """
    For each arm label in `arms`, call the LLM to get days-to-initiation predictions.
    Returns a DataFrame of predictions with same index as df.
    """
    preds = {key: [] for key in arms}
    for _, row in df.iterrows():
        for col_key, label in arms.items():
            prompt = generate_prompt(row, label)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            preds[col_key].append(parse_numeric(response.choices[0].message.content))
    return pd.DataFrame(preds, index=df.index)

# ─── Main Execution ───────────────────────────────────────────────────────────
# Load data
df = pd.read_csv(INPUT_CSV)

# Initialize LLM clients from environment variables
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# Define mapping of output columns to human-readable arm labels
openai_arms = {
    "chat_SOC_days":   "Standard of care",
    "chat_HIV_days":   "HIV screening",
    "chat_HIVTB_days": "HIV + TB screening"
}

# Generate OpenAI predictions
print("🔍 Generating OpenAI days-to-initiation predictions...")
openai_preds = get_predictions(chat_client, OPENAI_MODEL, df, openai_arms)

# Prepare Deepseek column keys
deepseek_arms = {k.replace("chat_", "ds_"): v for k, v in openai_arms.items()}

# Generate Deepseek predictions
print("🔍 Generating Deepseek days-to-initiation predictions...")
deepseek_preds = get_predictions(ds_client, DEEPSEEK_MODEL, df, deepseek_arms)

# Combine predictions with original data
df_out = pd.concat([df, openai_preds, deepseek_preds], axis=1)

# Save to CSV
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved counterfactual predictions to {OUTPUT_CSV}")
 

