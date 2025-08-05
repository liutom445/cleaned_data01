import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV    = "trial14/trial14.csv"
OUTPUT_CSV   = "trial14/trial14_counterfactuals_updated.csv"
OPENAI_MODEL = "o4-mini"  # ChatGPT model

# Map the CSV’s Treatment values to nice descriptions
ARM_DESCRIPTIONS = {
    "No Active Rehabilitation": "usual care without a structured exercise programme",
    "TERECO":                     "6-week home-based TERECO exercise programme"
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def bool_str(val: int) -> str:
    return "Yes" if val == 1 else "No"

def generate_base_prompt(row: pd.Series, arm: str) -> str:
    """Build baseline-covariate narrative including the arm’s meaning."""
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
        f"Assigned intervention: **{arm}** ({ARM_DESCRIPTIONS[arm]}).\n"
    )

def generate_cf_prompt(row: pd.Series) -> str:
    """
    Combine the baseline narrative with the observed 6-week change
    and ask the counterfactual question for the other arm.
    """
    actual_arm = row["Treatment"]
    # There are only two arms in ARM_DESCRIPTIONS
    other_arm  = next(a for a in ARM_DESCRIPTIONS if a != actual_arm)

    # Compute the actual change from baseline to 6 weeks
    actual_change = row["YP_6MWD_6w"] - row["X_6MWD_0w"]

    base = generate_base_prompt(row, actual_arm)
    return (
        base +
        f"The patient as assigned to **{actual_arm}** had a change in 6-minute walking distance of "
        f"**{actual_change:.1f} meters** from baseline to 6 weeks.\n"
        f"What would their change have been if they instead received **{other_arm}** "
        f"({ARM_DESCRIPTIONS[other_arm]})?\n"
        "Reply with only the numeric change in meters (e.g., 35.0)."
    )

# ─── Main counterfactual loop ─────────────────────────────────────────────────
def main():
    # 1) Load & filter
    df = pd.read_csv(INPUT_CSV)
    df = df[df["Treatment"].isin(ARM_DESCRIPTIONS)].reset_index(drop=True)

    # 2) Initialize ChatGPT client
    client = OpenAI(api_key="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")


    total       = len(df)
    cf_preds    = []
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries for TERECO vs. No Active Rehabilitation...")

    # 3) Loop over each patient
    for idx, row in df.iterrows():
        system_msg  = {"role": "system", "content": "You are an expert in pulmonary rehabilitation."}
        user_prompt = generate_cf_prompt(row)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        cf_val = parse_numeric(resp.choices[0].message.content)
        cf_preds.append(cf_val)
        query_count += 1

        # Progress & ETA every 30 calls (or at end)
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 4) Save only the CF column
    df[f"{OPENAI_MODEL}_cf_6MWD_6w"] = cf_preds
    df[[f"{OPENAI_MODEL}_cf_6MWD_6w"]].to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
