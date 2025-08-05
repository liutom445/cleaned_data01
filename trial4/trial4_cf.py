import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV    = "trial4/trial4.csv"                # Path to your Trial 4 data
OUTPUT_CSV   = "trial4/trial4_cf_counterfactuals_updated.csv"
MODEL        = "gpt-4.1-mini"              # ChatGPT model of your choice

# ─── Covariate definitions (Sriphoosanaphan et al.) ───────────────────────────
COV_DEFS = {
    'X_FIB4_0w': 'Fibrosis-4 score at baseline (non-invasive fibrosis index)',
    'X_APRI_0w': 'AST-to-platelet ratio index at baseline (fibrosis marker)',
    'X_VD_0w':   'Serum 25-hydroxyvitamin D at baseline (ng/mL)',
    'X_AST_0w':  'Aspartate aminotransferase at baseline (U/L)',
    'X_ALT_0w':  'Alanine aminotransferase at baseline (U/L)',
    'X_Plt_0w':  'Platelet count at baseline (10^3 cells/µL)',
    'X_TGF_0w':  'TGF-β1 at baseline (ng/mL)',
    'X_TIMP_0w': 'TIMP-1 at baseline (ng/mL)',
    'X_MMP_0w':  'MMP-9 at baseline (ng/mL)',
    'X_P3NP_0w': 'P3NP at baseline (ng/mL)'
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first float-like token, or return NaN if none."""
    txt = text.strip()
    if txt.lower() == "nan":
        return float("nan")
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", txt)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_cf_prompt(row: pd.Series) -> str:
    """
    Build a counterfactual prompt for Trial 4:
    - Lists baseline covariates with definitions
    - States the actual observed ΔP3NP at 6 weeks
    - Asks for the counterfactual under the other arm
    """
    # Map your Treatment codes to human labels
    actual_arm = row["Treatment"]
    if actual_arm == "VD":
        actual_label = "Vitamin D₂"
        other_label  = "Placebo"
    else:
        actual_label = "Placebo"
        other_label  = "Vitamin D₂"

    # Build prompt
    lines = [
        "Disclaimer: This is for educational simulation only, not medical advice.",
        "",
        "You are an expert in hepatology and liver-fibrosis biomarkers.",
        "",
        "Patient baseline characteristics (covariate = value):"
    ]
    for cov, desc in COV_DEFS.items():
        lines.append(f"  • {cov} = {row[cov]} ({desc})")

    lines += [
        "",
        f"The patient as assigned to **{actual_label}** had a 6-week change in P3NP of "
        f"**{row['YP_delta_P3NP_6w']:.2f} ng/mL**.",
        f"What would their 6-week change in P3NP have been if they instead received **{other_label}**?",
        "",
        "Reply with only the numeric change in ng/mL (e.g., -0.50)."
    ]
    return "\n".join(lines)


# ─── Main CF routine ───────────────────────────────────────────────────────────
def main():
    # 1) Load & filter to valid rows
    df = (
        pd.read_csv(INPUT_CSV)
          .dropna(subset=["Treatment", "YP_delta_P3NP_6w"])
          .reset_index(drop=True)
    )
    # Keep only the two arms
    df = df[df["Treatment"].isin(["VD", "Placebo"])].reset_index(drop=True)

    # 2) Initialize ChatGPT client
    client = OpenAI(api_key="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")


    # 3) Prepare storage & timing
    total       = len(df)
    cf_preds    = []
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries (model={MODEL})...")

    # 4) Loop and query
    for i, row in df.iterrows():
        system_msg  = {
            "role": "system",
            "content": "You are an expert in hepatology and liver-fibrosis biomarkers."
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

        # Progress & ETA every 30 or last
        if (i + 1) % 30 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / (i + 1)) * (total - (i + 1))
            print(f"🛎️ {i+1}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 5) Save results
    df["chatgpt_cf_P3NP_6w"] = cf_preds
    df[["chatgpt_cf_P3NP_6w"]].to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Counterfactuals saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
