import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV    = "trial36.csv"      # Path to input CSV with baseline covariates
OUTPUT_DIR   = "trial36_results/"  # Directory to save counterfactual CSVs
OPENAI_MODEL = "gpt-4.1"           # LLM model name for predictions

# ─── Define treatment-control contrast ───────────────────────────────────────
ARMS = {
    "ACI": "Active Control Intervention: 16 group psychoeducation sessions",
    "PI":  "Partial Intervention: tramiprosate 100 mg/day + 4 Mediterranean-diet classes"
}

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value embedded in an LLM response."""
    match = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not match:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(match.group(0))


def generate_outcome_prompt(row: pd.Series, intervention_label: str) -> str:
    """
    Constructs a prompt for predicting the 12-month change in the attention–executive composite z-score.

    COVARIATES (baseline, month 0):
      • X_GDS_0m: Geriatric Depression Scale score (0–15)
      • X_EMQ_0m: Everyday Memory Questionnaire score
      • X_STAI_0m: State–Trait Anxiety Inventory score
      • X_MMSE_0m: Mini–Mental State Examination score (0–30)
      • X_RAVLT_delayedrecall_0m: RAVLT delayed-recall score (number correct)
      • X_IPAQ_total_0m: IPAQ total physical activity (MET-min/week)
      • X_MeDi_score_0m: Mediterranean-diet adherence score (0–9)
      • X_MemoryNew_0m: Memory composite z-score
      • X_AttentionExe_0m: Attention–Executive composite z-score
      • X_VisuoSpNew_0m: Visuo-spatial composite z-score
    """
    prompt = [
        "Disclaimer: This is an educational simulation only.\n\n",
        "You are an expert in cognitive interventions and clinical trial outcomes.\n",
        "Predict the change in the attention–executive composite z-score from baseline to 12 months.\n",
        "Provide only a single numeric value.\n\n",
        "Patient baseline data:\n",
        f"  • Intervention assigned: {intervention_label}\n",
        f"  • Geriatric Depression Scale (X_GDS_0m): {row['X_GDS_0m']}\n",
        f"  • Everyday Memory Questionnaire (X_EMQ_0m): {row['X_EMQ_0m']}\n",
        f"  • State–Trait Anxiety Inventory (X_STAI_0m): {row['X_STAI_0m']}\n",
        f"  • MMSE score (X_MMSE_0m): {row['X_MMSE_0m']}\n",
        f"  • RAVLT delayed recall (X_RAVLT_delayedrecall_0m): {row['X_RAVLT_delayedrecall_0m']}\n",
        f"  • IPAQ total MET-min/week (X_IPAQ_total_0m): {row['X_IPAQ_total_0m']}\n",
        f"  • Mediterranean diet score (X_MeDi_score_0m): {row['X_MeDi_score_0m']}\n",
        f"  • Memory composite z-score (X_MemoryNew_0m): {row['X_MemoryNew_0m']}\n",
        f"  • Attention–Executive composite z-score (X_AttentionExe_0m): {row['X_AttentionExe_0m']}\n",
        f"  • Visuo-spatial composite z-score (X_VisuoSpNew_0m): {row['X_VisuoSpNew_0m']}\n"
    ]
    return ''.join(prompt)


def predict_counterfactuals(input_csv: str, output_dir: str, api_key: str, model: str = OPENAI_MODEL):
    # Load baseline covariates
    df = pd.read_csv(input_csv)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Prepare result columns for each arm
    for arm in ARMS:
        df[f"{model}_AttentionExe_12m_{arm}"] = None

    # Progress tracking
    total_queries = len(df) * len(ARMS)
    start_time    = time.time()
    query_count   = 0
    print(f"Starting {total_queries} LLM queries for ACI vs PI...")

    # Iterate and query for each arm
    for idx, row in df.iterrows():
        sys_msg = {"role": "system", "content": "You are an expert in clinical trial outcome prediction."}
        for arm_code, arm_label in ARMS.items():
            prompt = generate_outcome_prompt(row, arm_label)
            resp   = client.chat.completions.create(model=model, messages=[sys_msg, {"role": "user", "content": prompt}])
            df.at[idx, f"{model}_ΔAttentionExe_12m_{arm_code}"] = parse_numeric(resp.choices[0].message.content)
            query_count += 1

            # Print progress and ETA every 30 queries
            if query_count % 30 == 0 or query_count == total_queries:
                elapsed = time.time() - start_time
                eta     = (elapsed / query_count) * (total_queries - query_count)
                print(f"🛎️ {query_count}/{total_queries} queries | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_path  = os.path.join(output_dir, f"{base_name}_{model}_counterfactuals.csv")

    # Save results
    df.to_csv(out_path, index=False)
    print(f"✅ Counterfactual predictions saved to {out_path}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate counterfactual attention–executive composite predictions for Trial 36")
    parser.add_argument(
        "--input", default="trial36.csv",
        help="Path to input CSV with baseline covariates"
    )
    parser.add_argument(
        "--prefix", dest="output_dir",
        default="trial36/",
        help="Directory prefix for output files"
    )
    parser.add_argument(
        "--api_key",default = "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",
        help="OpenAI API key for authentication"
    )
    parser.add_argument(
        "--model", default=OPENAI_MODEL,
        help="LLM model name to use for predictions"
    )
    args = parser.parse_args()

    predict_counterfactuals(
        input_csv=args.input,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model
    )
