import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = "trial6/trial6.csv"
OUTPUT_CSV = "trial6/trial6_counterfactuals.csv"
MODEL      = "gpt-4.1-mini"  # ChatGPT model for predictions

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """
    Extract the first numeric value from an LLM response.
    If the response is exactly 'nan' (case-insensitive), return float('nan').
    Otherwise, try regex; if that fails, raise ValueError.
    """
    txt = text.strip()
    if txt.lower() == "nan":
        return float("nan")

    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", txt)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_prompt(row: pd.Series, treatment: str) -> str:
    """Build the baseline‐covariate narrative including assigned treatment."""
    sex    = "Male" if row['X_sex_0m'] == 1 else "Female"
    statin = "Yes"  if row['X_statin_0m'] == 1 else "No"

    return (
        "Disclaimer: Simulation for educational purposes.\n\n"
        "You are an expert in diabetes and cardiovascular research.\n"
        "Patient baseline characteristics:\n"
        f"- Age: {row['X_age_0m']} years\n"
        f"- Sex: {sex}\n"
        f"- Statin use at baseline: {statin}\n"
        f"- Pre-randomization diabetes therapy: {row['X_dm_treatment_0m']}\n"
        f"- Baseline systolic BP: {row['X_sbp_0m']} mmHg\n"
        f"- Baseline HbA1c: {row['X_hba1c_ancova_0m']} %\n"
        f"- Baseline maximum CCA IMT: {row['X_max_imt_0m']} mm\n\n"
        f"Assigned treatment: **{treatment}**.\n"
    )


def generate_cf_prompt(row: pd.Series) -> str:
    """Combine baseline narrative with observed outcome and the CF question."""
    actual_arm = row["Treatment"]                          # "Sitagliptin" or "Conventional"
    other_arm  = "Conventional" if actual_arm == "Sitagliptin" else "Sitagliptin"
    actual_val = row["YP_delta_CCA_IMT_24m"]

    base = generate_prompt(row, actual_arm)
    return (
        base +
        f"The patient as assigned to **{actual_arm}** had a 24-month change in mean CCA IMT of **{actual_val:.2f} mm**.\n"
        f"What would their 24-month change have been if they instead received **{other_arm}**?\n"
        "Reply with only the numeric value in millimeters (e.g., 0.05)."
    )


# ─── Main counterfactual routine ──────────────────────────────────────────────
def main():
    # 1) Load & filter to the two arms
    df = pd.read_csv(INPUT_CSV)
    df = df[df["Treatment"].isin(["Sitagliptin", "Conventional"])].reset_index(drop=True)

    # 2) Initialize ChatGPT client
    client = OpenAI(api_key="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")


    total       = len(df)
    cf_preds    = []
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries (ChatGPT {MODEL})...")

    # 3) Loop over each patient
    for idx, row in df.iterrows():
        system_msg  = {"role": "system", "content": "You are an expert in diabetes and cardiovascular research."}
        user_prompt = generate_cf_prompt(row)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        content = resp.choices[0].message.content

        # 4) Parse numeric, but catch & log any failures
        try:
            cf_val = parse_numeric(content)
        except ValueError:
            print(f"⚠️ Warning: Non-numeric response for row {idx}: {content!r}. Setting to NaN.")
            cf_val = float("nan")

        cf_preds.append(cf_val)
        query_count += 1

        # 5) Progress & ETA every 30 calls (or at end)
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 6) Save only the CF column
    df[f"{MODEL}_cf_CCA_IMT_24m"] = cf_preds
    df[[f"{MODEL}_cf_CCA_IMT_24m"]].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
