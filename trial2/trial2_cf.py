import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = "trial2/trial2.csv"                  # Path to Trial 2 data
OUTPUT_CSV = "trial2/trial2_counterfactuals_updated.csv"  # Where to save CF predictions
MODEL      = "gpt-4.1-mini"                     # LLM model to use

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """
    Extract the first numeric value from an LLM response.
    If the response is exactly 'nan', return NaN.
    """
    txt = text.strip()
    if txt.lower() == "nan":
        return float("nan")
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", txt)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_cf_prompt(row: pd.Series) -> str:
    """
    Build a counterfactual prompt for recovery time:
     - List baseline covariates in human‐readable form
     - State the observed recovery time under the assigned treatment
     - Ask for the recovery time if they had received the other arm
    """
    # Map treatment codes to human labels
    actual = row["Treatment"]
    if actual == "Ivermectin+Doxycycline":
        actual_lbl = "Ivermectin plus Doxycycline"
        other_lbl  = "Placebo"
    else:
        actual_lbl = "Placebo"
        other_lbl  = "Ivermectin plus Doxycycline"

    # Human‐friendly covariate strings
    sex      = "Male" if row["X_sex_0d"] == 1 else "Female"
    fever    = "Yes"  if row["X_fever_0d"] == 1 else "No"
    cough    = "Yes"  if row["X_cough_0d"] == 1 else "No"
    respdiff = "Yes"  if row["X_respdiff_0d"] == 1 else "No"
    comorb   = "Yes"  if row["X_comorb_0d"] == 1 else "No"
    diabetes = "Yes"  if row["X_diabetes_0d"] == 1 else "No"
    hyperten = "Yes"  if row["X_hyperten_0d"] == 1 else "No"

    lines = [
        "Disclaimer: This is an educational simulation only, not medical advice.\n",
        "You are a clinical trial expert predicting time to recovery from COVID-19.\n",
        "Patient baseline characteristics:",
        f"- Age group: {row['X_agegrp_0d']}",
        f"- Sex: {sex}",
        f"- Fever at baseline: {fever}",
        f"- Cough at baseline: {cough}",
        f"- Respiratory difficulty at baseline: {respdiff}",
        f"- Any comorbidity: {comorb}",
        f"- Diabetes: {diabetes}",
        f"- Hypertension: {hyperten}\n",
        f"The patient as assigned to **{actual_lbl}** had a time to recovery of **{row['YP_recovery_time']:.1f} days**.\n",
        f"What would their time to recovery have been if they instead received **{other_lbl}**?\n",
        "Please reply with only the number of days (e.g., 7.0)."
    ]
    return "\n".join(lines)


def main():
    # 1) Load & filter to valid rows
    df = (
        pd.read_csv(INPUT_CSV)
          .dropna(subset=["Treatment", "YP_recovery_time"])
          .query("Treatment in ['Ivermectin+Doxycycline', 'Placebo']")
          .reset_index(drop=True)
    )

    # 2) Initialize OpenAI client
    client = OpenAI(api_key="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")


    total    = len(df)
    cf_preds = []
    start    = time.time()

    print(f"Starting {total} counterfactual queries (model={MODEL})...")

    # 3) Loop over each patient
    for i, row in df.iterrows():
        system_msg  = {
            "role": "system",
            "content": "You are a clinical trial expert predicting recovery times."
        }
        user_prompt = generate_cf_prompt(row)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        content = resp.choices[0].message.content

        try:
            cf_val = parse_numeric(content)
        except ValueError:
            print(f"⚠️ Row {i} non-numeric response: {content!r} → setting NaN")
            cf_val = float("nan")

        cf_preds.append(cf_val)

        # Progress & ETA every 30 or at end
        if (i + 1) % 30 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            eta     = (elapsed / (i + 1)) * (total - (i + 1))
            print(f"🛎️ {i+1}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 4) Save results
    df["chatgpt_cf_recovery_time"] = cf_preds
    df[["chatgpt_cf_recovery_time"]].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
