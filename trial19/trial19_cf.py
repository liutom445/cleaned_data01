import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY     = "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"
MODEL       = "o4-mini"
INPUT_CSV   = "trial19/trial19.csv"
OUTPUT_CSV  = "trial19/trial19_cf.csv"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_cf_prompt(row: pd.Series, other_label: str) -> str:
    """Build a prompt asking for the counterfactual portal gradient."""
    lines = [
        "Disclaimer: This is an educational simulation only and not medical advice.",
        "",
        "You are an expert in transplant hemodynamics interpreting patient data.",
        "",
        "Patient baseline covariates:",
        f"  • Age (years): {int(row['X_RECIPIENT_AGE_YEARS_0d'])}",
        f"  • Sex: {row['X_RECIPIENT_GENDER_0d']}",
        f"  • Diagnosis: {row['X_RECIPIENT_DIAGNOSIS_0d']}",
        f"  • Child-Pugh class: {row['X_CHILD_PUGH_CLASS_0d']}",
        f"  • MELD score: {row['X_MELD_SCORE_0d']}",
        f"  • Baseline creatinine (mg/dL): {row['X_CREATININE_0d']:.2f}",
        f"  • Donor age (years): {int(row['X_DONOR_AGE_YEARS_0d'])}",
        f"  • Donor sex: {row['X_DONOR_GENDER_0d']}",
        f"  • Donor diagnosis: {row['X_DONOR_DIAGNOSIS_0d']}",
        f"  • Graft sharing level: {row['X_GRAFT_SHARING_0d']}",
        f"  • Surgical technique: {row['Treatment']}",
        "",
        # Actual observed outcome
        f"The patient as treated ({row['Treatment']}) had a portal venous pressure gradient of "
        f"{row['YP_FHVP_CVP_GRADIENT']:.1f} mm Hg.",
        f"What would their portal venous pressure gradient be if they instead received {other_label}?",
        "",
        "Answer with only the numeric value in mm Hg."
    ]
    return "\n".join(lines)

def main():
    # 1) Initialize OpenAI client
    client = OpenAI(api_key=API_KEY)

    # 2) Load and clean data
    df = (
        pd.read_csv(INPUT_CSV)
          .dropna(subset=["YP_FHVP_CVP_GRADIENT"])
          .reset_index(drop=True)
    )

    # 3) Prepare for counterfactual calls
    cf_preds    = []
    total       = len(df)
    query_count = 0
    start_time  = time.time()

    # 4) Loop over each patient
    for idx, row in df.iterrows():
        # Determine the “other” arm
        other_label = ("Conventional" 
                       if row["Treatment"] == "Piggyback" 
                       else "Piggyback")

        # Build & send the ChatGPT prompt
        system_msg  = {"role": "system", "content": "You are an expert in transplant hemodynamics."}
        user_prompt = generate_cf_prompt(row, other_label)
        resp        = client.chat.completions.create(
                          model=MODEL,
                          messages=[system_msg, {"role":"user","content":user_prompt}]
                      )

        # Parse and store
        cf_value    = parse_numeric(resp.choices[0].message.content)
        cf_preds.append(cf_value)
        query_count += 1

        # Progress & ETA every 30 calls
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} queries | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 5) Save results
    df["chatgpt_cf_gradient"] = cf_preds
    df[["chatgpt_cf_gradient"]].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved counterfactual predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
