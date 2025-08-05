import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV       = "trial17/trial17.csv"
OUTPUT_CSV      = "trial17/trial17_counterfactuals_updated.csv"
MODEL           = "gpt-4.1"
CONTROL_ARM     = "statin"         # rosuvastatin 20 mg daily
TREATMENT_ARM   = "statin_pcsk9"   # rosuvastatin 10 mg daily + alirocumab 75 mg Q2W

# Human‐readable descriptions for prompt clarity
ARM_DESCRIPTIONS = {
    CONTROL_ARM:   "rosuvastatin 20 mg once daily",
    TREATMENT_ARM: "rosuvastatin 10 mg once daily + alirocumab 75 mg every 2 weeks"
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build a clear baseline‐covariate narrative including treatment meaning."""
    sex   = "Male"   if row['X_sex_0w'] == 1 else "Female"
    hbp   = "Yes"    if row['X_hbp_0w'] == 1 else "No"
    smoke = "Yes"    if row['X_smoke_0w'] == 1 else "No"
    dm    = "Yes"    if row['X_DM_0w']   == 1 else "No"

    return (
        "Disclaimer: Simulation for educational purposes.\n\n"
        "You are an expert in lipid‐lowering therapies.\n"
        "Patient baseline characteristics:\n"
        f"- Age: {row['X_age_0w']} years\n"
        f"- Sex: {sex}\n"
        f"- Hypertension history: {hbp}\n"
        f"- Smoking history: {smoke}\n"
        f"- Diabetes history: {dm}\n"
        f"- Body mass index: {row['X_BMI_0w']:.1f} kg/m²\n"
        f"- Baseline LDL‐C: {row['X_LDL_0w']} mg/dL\n"
        f"- Baseline total cholesterol: {row['X_TC_0w']} mg/dL\n"
        f"- Baseline triglycerides: {row['X_TG_0w']} mg/dL\n"
        f"- Baseline HDL‐C: {row['X_HDL_0w']} mg/dL\n\n"
        f"Assigned treatment: **{treatment}** "
        f"({ARM_DESCRIPTIONS[treatment]}).\n"
    )

def generate_cf_prompt(row: pd.Series) -> str:
    """
    Build the counterfactual prompt: baseline narrative + actual outcome
    + question about the opposite arm (with description).
    """
    actual_arm = row["Treatment"]
    other_arm  = CONTROL_ARM if actual_arm == TREATMENT_ARM else TREATMENT_ARM
    actual_y   = row["YP_delta_LDL_24w"]

    base = generate_prompt(row, actual_arm)
    return (
        base +
        f"The patient as assigned to **{actual_arm}** had a 24-week percent change in LDL‐C of **{actual_y:.2f}%**.\n"
        f"What would their percent change in LDL‐C have been if they instead received **{other_arm}** "
        f"({ARM_DESCRIPTIONS[other_arm]})?\n"
        "Reply with only the numeric percentage (e.g., -12.3)."
    )

# ─── Main CF loop ─────────────────────────────────────────────────────────────
def main():
    # 1) Load & filter to only our two arms
    df = pd.read_csv(INPUT_CSV)
    df = df[df["Treatment"].isin([CONTROL_ARM, TREATMENT_ARM])].reset_index(drop=True)

    # 2) Init ChatGPT client
    client = OpenAI(api_key="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")

    # 3) Prepare iteration
    total       = len(df)
    cf_preds    = []
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries for {CONTROL_ARM} vs. {TREATMENT_ARM}...")

    # 4) Loop rows
    for idx, row in df.iterrows():
        system_msg  = {"role": "system", "content": "You are an expert in lipid‐lowering therapies."}
        user_prompt = generate_cf_prompt(row)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )

        cf_val = parse_numeric(resp.choices[0].message.content)
        cf_preds.append(cf_val)
        query_count += 1

        # Progress & ETA
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 5) Save results
    df[f"{MODEL}_cf_LDLpct_24w"] = cf_preds
    df[[f"{MODEL}_cf_LDLpct_24w"]].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
