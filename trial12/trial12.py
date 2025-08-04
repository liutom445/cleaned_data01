import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV    = "trial12.csv"  # Path to input CSV with baseline covariates
OUTPUT_DIR   = "trial12_results/"  # Directory to save counterfactual CSVs
OPENAI_MODEL = "o4-mini"         # LLM model name for predictions

# ─── Define review process contrast ──────────────────────────────────────────
ARMS = {
    "SingleBlind": "Single-blind peer review (author identities known to reviewers)",
    "DoubleBlind": "Double-blind peer review (author identities anonymized to reviewers)"
}

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value embedded in an LLM response."""
    match = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not match:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(match.group(0))


def generate_outcome_prompt(row: pd.Series, review_type: str) -> str:
    """
    Constructs a prompt for predicting the reviewer rating for a manuscript under the specified review type.

    COVARIATE CATEGORIES:
      • Submission year (X_Year_0w): year of manuscript submission
      • Author gender (X_AuthorGender_0w): 1=Male, 2=Female
      • Author HDI category (X_AuthorHDICat_0w): 1=Very high, 2=High, 3=Medium, 4=Low
      • Author English proficiency (X_AuthorEnglish_0w): 0=Non-proficient, 1=Proficient
    """
    # Map codes to labels
    gender_map = {1: 'Male', 2: 'Female'}
    hdi_map = {1: 'Very high HDI', 2: 'High HDI', 3: 'Medium HDI', 4: 'Low HDI'}
    english_map = {0: 'Non-proficient', 1: 'Proficient'}

    prompt = [
        "Disclaimer: This is an educational simulation only.\n\n",
        "You are an expert in academic peer review processes.\n",
        f"Predict the reviewer rating for this manuscript if reviewed under the following process: {review_type}.\n",
        "Provide only a single numeric value (e.g., on a 1–5 scale).\n\n",
        "Manuscript baseline data and covariates:\n",
        f"  • Submission year (X_Year_0w): {row['X_Year_0w']}\n",
        f"  • Author gender: {gender_map.get(row['X_AuthorGender_0w'], 'Unknown')} ({row['X_AuthorGender_0w']})\n",
        f"  • Author HDI category: {hdi_map.get(row['X_AuthorHDICat_0w'], 'Unknown')} ({row['X_AuthorHDICat_0w']})\n",
        f"  • Author English proficiency: {english_map.get(row['X_AuthorEnglish_0w'], 'Unknown')} ({row['X_AuthorEnglish_0w']})\n",
    ]
    return ''.join(prompt)


def predict_counterfactuals(input_csv: str, output_dir: str, api_key: str, model: str = OPENAI_MODEL):
    # Load manuscript data
    df = pd.read_csv(input_csv)

    # Initialize LLM client
    client = OpenAI(api_key=api_key)

    # Prepare result columns for each review type
    for arm_code in ARMS:
        df[f"{model}_ReviewScore_{arm_code}"] = None

    # Progress tracking
    total_queries = len(df) * len(ARMS)
    start_time    = time.time()
    query_count   = 0
    print(f"Starting {total_queries} LLM queries for trial12 review-rating predictions...")

    # Iterate and collect predictions
    for idx, row in df.iterrows():
        system_msg = {"role": "system", "content": "You are an expert in academic peer review processes."}
        for arm_code, arm_label in ARMS.items():
            prompt = generate_outcome_prompt(row, arm_label)
            resp   = client.chat.completions.create(model=model, messages=[system_msg, {"role": "user", "content": prompt}])
            df.at[idx, f"{model}_ReviewScore_{arm_code}"] = parse_numeric(resp.choices[0].message.content)
            query_count += 1

            # Print progress and ETA every 30 queries
            if query_count % 30 == 0 or query_count == total_queries:
                elapsed = time.time() - start_time
                eta     = (elapsed / query_count) * (total_queries - query_count)
                print(f"🛎️ {query_count}/{total_queries} queries | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_path  = os.path.join(output_dir, f"{base_name}_{model}_review_predictions.csv")

    # Save results
    df.to_csv(out_path, index=False)
    print(f"✅ Counterfactual review-rating predictions saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate counterfactual reviewer ratings for Trial 12")
    parser.add_argument(
        "--input", default="trial12.csv",
        help="Path to input CSV with manuscript covariates"
    )
    parser.add_argument(
        "--prefix", dest="output_dir",
        default="trial 12/",
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
