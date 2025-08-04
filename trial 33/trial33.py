import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV      = "trial33.csv"
OUTPUT_DIR     = "trial33_results/"   # directory prefix for output files
OPENAI_MODEL   = "gpt-4.1"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_outcome_prompt(row, treatment_label):
    """
    Build a prompt asking for the expected change in the primary outcome
    after 6 months under the specified intervention for this patient.

    COVARIATE CODES:
      • Gender: 1 = Male, 2 = Female
      • Ethnicity: 0 = Non-Hispanic, 1 = Hispanic
      • Race: 1 = White, 2 = Black, 3 = Asian, 6 = Other
      • Education: 0 = High school or less, 1 = Bachelor's degree, 2 = Graduate degree
      • Employment: 1 = Employed, 2 = Unemployed/Retired (0 = missing)
      • Disease Group: 1 = Diabetes, 2 = Hypertension, 4 = Head/Neck Cancer,
                       5 = Breast Cancer, 6 = Major Depression
      • Volunteer: 0 = No prior research/health volunteering, 1 = Has volunteered
    """
    return (
        "Disclaimer: This is an educational simulation only.\n\n"
        "You are an expert in clinical trial outcome prediction.\n"
        "Use the following patient baseline data and code definitions carefully.\n\n"
        "Baseline covariates:\n"
        f"  • Treatment arm: {treatment_label} (Reframing vs. Standard)\n"
        f"  • Therapeutic Misconception score at baseline: {row['YP_TM_total_score']}\n"
        f"  • Willingness to participate at baseline: {row['YS_willing_participate']}\n"
        f"  • Age (years): {row['X_age_0w']}\n"
        f"  • Gender code: {row['X_Gender_0w']}\n"
        f"  • Ethnicity code: {row['X_Ethnicity_0w']}\n"
        f"  • Race code: {row['X_Race_0w']}\n"
        f"  • Education code: {row['X_Education_0w']}\n"
        f"  • Employment code: {row['X_Employment_0w']}\n"
        f"  • Disease Group code: {row['X_DiseaseGroup_0w']}\n"
        f"  • Prior volunteering code: {row['X_Volunteer_0w']}\n\n"
        "Predict the new primary outcome (Therapeutic Misconception score) "
        "from baseline to 6 months, expressed as a single numeric value. "
        "Answer with only the number."
    )


def predict_counterfactuals(input_csv, output_dir, api_key, model=OPENAI_MODEL):
    # Load data
    df = pd.read_csv(input_csv)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Prepare output columns
    cols = [f"{model}_TMscore_6m_R", f"{model}_TMscore_6m_S"]
    df[cols] = None

    # Progress tracking
    total_queries = len(df) * 2  # treatment + control per row
    start_time    = time.time()
    query_count   = 0
    print(f"Starting {total_queries} LLM queries...")

    # Iterate over each patient record
    for idx, row in df.iterrows():
        system_msg = {"role": "system", "content": "You are an expert in clinical trial outcome prediction."}

        # Predict under Reframing (treatment)
        prompt_r = generate_outcome_prompt(row, treatment_label="Reframing")
        resp_r   = client.chat.completions.create(model=model, messages=[system_msg, {"role": "user", "content": prompt_r}])
        df.at[idx, cols[0]] = parse_numeric(resp_r.choices[0].message.content)
        query_count += 1

        # Predict under Standard (control)
        prompt_s = generate_outcome_prompt(row, treatment_label="Standard")
        resp_s   = client.chat.completions.create(model=model, messages=[system_msg, {"role": "user", "content": prompt_s}])
        df.at[idx, cols[1]] = parse_numeric(resp_s.choices[0].message.content)
        query_count += 1

        # Print progress and ETA every 30 queries
        if query_count % 30 == 0 or query_count == total_queries:
            elapsed   = time.time() - start_time
            avg_time  = elapsed / query_count
            remaining = total_queries - query_count
            eta       = remaining * avg_time
            print(f"🛎️ {query_count}/{total_queries} queries | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_csv))[0]}_{model}_counterfactuals.csv")

    # Save results
    df.to_csv(output_path, index=False)
    print(f"✅ Counterfactual predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate counterfactual Therapeutic Misconception score predictions"
    )
    parser.add_argument(
        "--input", 
        default="Research/collection/SS 25/Meeting 0729/cleaned_data/trial 33/trial33.csv",
        help="Path to input CSV with baseline covariates"
    )
    parser.add_argument(
        "--prefix",
        dest="output_dir",
        default="Research/collection/SS 25/Meeting 0729/cleaned_data/trial 33/",
        help="Prefix (directory) for output files"
    )
    parser.add_argument(
        "--api_key",
        default="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",
        help="OpenAI API key for authentication"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="LLM model name to use for predictions"
    )
    args = parser.parse_args()

    predict_counterfactuals(
        input_csv=args.input,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model
    )
