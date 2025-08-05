import os
import re
import time
import argparse
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV    = "trial 36/trial36.csv"
OUTPUT_DIR   = "trial36/"
MODEL        = "o4-mini"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_cf_prompt(row: pd.Series) -> str:
    """
    Build a prompt for the counterfactual 12-month attention–executive change,
    using the actual observed change and assigned treatment.
    """
    # Determine arms
    actual_arm = row["Treatment"]                      # "PI" or "MI"
    other_arm  = "MI" if actual_arm == "PI" else "PI"
    actual_val = row["YP_delta_AttentionExe_12m"]

    lines = [
        "Disclaimer: This is an educational simulation only.",
        "",
        "You are an expert in cognitive interventions and clinical trial outcomes.",
        "",
        "Baseline covariates (month 0):",
        f"  • IPAQ total MET-min/week:    {row['X_IPAQ_total_0m']}",
        f"  • Mediterranean diet score:   {row['X_MeDi_score_0m']}",
        f"  • Memory composite z-score:    {row['X_MemoryNew_0m']}",
        f"  • Attention–Executive z-score: {row['X_AttentionExe_0m']}",
        f"  • Visuo-spatial z-score:       {row['X_VisuoSpNew_0m']}",
        "",
        f"The patient as assigned to **{actual_arm}** had a 12-month change in the attention–executive composite z-score of **{actual_val:.3f}**.",
        f"What would their 12-month attention–executive change be if they instead received **{other_arm}**?",
        "",
        "Answer with only the numeric change from baseline to 12 months."
    ]
    return "\n".join(lines)

def predict_counterfactuals(input_csv: str,
                            output_dir: str,
                            api_key: str,
                            model: str = MODEL):
    # 1) Load data
    df = pd.read_csv(input_csv)

    # 2) Init OpenAI client
    client = OpenAI(api_key=api_key)

    # 3) Prepare storage
    cf_preds    = []
    total       = len(df)
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual LLM queries...")

    # 4) Iterate rows
    for idx, row in df.iterrows():
        system_msg = {
            "role": "system",
            "content": "You are an expert in cognitive interventions and clinical trial outcomes."
        }
        user_prompt = generate_cf_prompt(row)
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        cf_val = parse_numeric(resp.choices[0].message.content)
        cf_preds.append(cf_val)

        query_count += 1
        # Progress & ETA every 30 calls
        if query_count % 30 == 0 or query_count == total:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total - query_count)
            print(f"🛎️ {query_count}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 5) Save only the CF column
    df[f"{model}_cf_AttentionExe_12m"] = cf_preds

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_path = os.path.join(output_dir, f"{base}_{model}_cf.csv")
    df[[f"{model}_cf_AttentionExe_12m"]].to_csv(out_path, index=False)

    print(f"✅ Saved counterfactual predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate counterfactual attention–executive predictions for Trial 36"
    )
    parser.add_argument(
        "--input",
        default=INPUT_CSV,
        help="Path to input CSV with baseline covariates"
    )
    parser.add_argument(
        "--prefix",
        dest="output_dir",
        default=OUTPUT_DIR,
        help="Directory for output files"
    )
    parser.add_argument(
        "--api_key",
        default="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",
        help="OpenAI API key (or rely on OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="LLM model name to use for predictions"
    )
    args = parser.parse_args()

    predict_counterfactuals(
        input_csv=args.input,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model
    )
