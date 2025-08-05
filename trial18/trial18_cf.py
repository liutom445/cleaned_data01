import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV    = "trial18/trial18.csv"
OUTPUT_CSV   = "trial18/trial18_counterfactuals_updated.csv"
MODEL        = "gpt-4.1"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_cf_prompt(row: pd.Series, actual_label: str, other_label: str) -> str:
    """
    Build a counterfactual prompt using the observed time-to-disengagement
    and baseline covariates, asking what it would have been under the other arm.
    """
    lines = [
        "Disclaimer: Simulation for educational purposes.\n",
        "You are an expert in HIV care retention interpreting patient data.\n",
        "Patient baseline data:",
        f"• Age category: {row['X_AgeCat_0m']}",
        f"• Gender: {row['X_Gender_0m']}",
        f"• Years since diagnosis: {row['X_TimeSinceDx_0m']}",
        f"• Years in clinic: {row['X_TimeSinceClinic_0m']}",
        f"• On ART: {row['X_ARTStatus_0m']}",
        f"• # aware of patient's status: {row['X_yourHIVTotal_0m']}",
        f"• Material support: {row['X_MaterialSupportMean_0m']}",
        f"• Internalized stigma: {row['X_StigInternal_0m']}\n",
        # Observed outcome
        f"The patient as assigned to {actual_label} had a time to disengagement of "
        f"{row['YP_time_to_disengaged_12m']:.0f} days within 12 months.",
        f"What would their time to disengagement be if they instead received {other_label}?",
        "\nReply with only a single numeric value (days)."
    ]
    return "\n".join(lines)

def main():
    # 1) Load data
    df = pd.read_csv(INPUT_CSV)

    # 2) Initialize OpenAI client (pull key from env)
    client = OpenAI(api_key="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")


    # 3) Determine the two treatment arms
    treatments = df["Treatment"].unique().tolist()
    if len(treatments) != 2:
        raise RuntimeError(f"Expected exactly 2 arms in 'Treatment', found: {treatments}")

    # 4) Prepare for counterfactual loop
    cf_preds    = []
    total       = len(df)
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries using {MODEL}...")

    # 5) Loop over patients
    for idx, row in df.iterrows():
        actual = row["Treatment"]
        other  = treatments[1] if actual == treatments[0] else treatments[0]

        system_msg  = {"role": "system", "content": "You are an expert in HIV care retention."}
        user_prompt = generate_cf_prompt(row, actual, other)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        cf_val = parse_numeric(resp.choices[0].message.content)
        cf_preds.append(cf_val)

        query_count += 1
        # Progress & ETA every 30 calls (or at end)
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    # 6) Save results
    col_name = f"{MODEL}_cf_time_to_disengaged_12m"
    df[col_name] = cf_preds
    df[[col_name]].to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Counterfactual predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

