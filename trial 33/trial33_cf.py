import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV    = "trial 33/trial33.csv"        # Must include YP_TM_delta_6m and Treatment columns
OUTPUT_DIR   = "trial33/"    # Directory to save counterfactual CSVs
OPENAI_MODEL = "gpt-4.1"            # ChatGPT model for CF predictions

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    match = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not match:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(match.group(0))


def generate_outcome_prompt(row: pd.Series,
                             treatment_label: str,
                             other_potential: float,
                             other_label: str) -> str:
    """
    Build prompt asking for the counterfactual 6-month Therapeutic Misconception score change
    under the specified intervention, given the observed change in the original arm.
    """
    lines = [
        "Disclaimer: This is an educational simulation only.",
        "",
        "You are an expert in clinical trial outcome prediction.",
        "Use the following patient baseline data carefully.",
        "",
        "Baseline covariates:",
        f"  • Treatment arm assigned: {other_label}",
        f"  • Therapeutic Misconception score at baseline: {row['YP_TM_total_score']}",
        f"  • Willingness to participate at baseline: {row['YS_willing_participate']}",
        f"  • Age (years): {row['X_age_0w']}",
        f"  • Gender code: {row['X_Gender_0w']}",
        f"  • Ethnicity code: {row['X_Ethnicity_0w']}",
        f"  • Race code: {row['X_Race_0w']}",
        f"  • Education code: {row['X_Education_0w']}",
        f"  • Employment code: {row['X_Employment_0w']}",
        f"  • Disease Group code: {row['X_DiseaseGroup_0w']}",
        f"  • Prior volunteering code: {row['X_Volunteer_0w']}",
        "",
        f"The patient assigned to {other_label} has an observed 6-month TM change of {other_potential}.",
        f"What would their 6-month TM change be if they instead received {treatment_label}?",
        "",
        "Predict the change in the Therapeutic Misconception score",
        "from baseline to 6 months, expressed as a single numeric value.",
        "Answer with only the number."
    ]
    return "\n".join(lines)


def predict_counterfactuals(input_csv: str,
                            output_dir: str,
                            api_key: str,
                            model: str = OPENAI_MODEL):
    # Load data
    df = pd.read_csv(input_csv)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Progress tracking
    total_queries = len(df)  # one CF query per patient
    start_time    = time.time()
    query_count   = 0
    print(f"Starting {total_queries} counterfactual LLM queries...")

    cf_list = []
    for idx, row in df.iterrows():
        system_msg = {"role": "system", "content": "You are an expert in clinical trial outcome prediction."}

        # Actual observed outcome and assignment
        actual_y = row['YP_TM_total_score']
        assigned = row['Treatment']  # 'Standard' or 'Reframing'

        # Determine labels for CF prompt
        if assigned == 'Standard':
            other_label     = 'Standard'
            treatment_label = 'Reframing'
        else:
            other_label     = 'Reframing'
            treatment_label = 'Standard'

        # Generate and send CF prompt
        prompt = generate_outcome_prompt(
            row,
            treatment_label=treatment_label,
            other_potential=actual_y,
            other_label=other_label
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg, {"role": "user", "content": prompt}]
        )
        cf_list.append(parse_numeric(resp.choices[0].message.content))
        query_count += 1

        # Progress logging every 30 queries or at end
        if query_count % 30 == 0 or query_count == total_queries:
            elapsed = time.time() - start_time
            eta     = (elapsed / query_count) * (total_queries - query_count)
            print(f"🛎️ {query_count}/{total_queries} queries | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # Save CF predictions
    df[f"{OPENAI_MODEL}_DTMscore_6m_cf"] = cf_list
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "trial33_chatgpt_counterfactual_tm.csv")
    df[[f"{OPENAI_MODEL}_DTMscore_6m_cf"]].to_csv(out_file, index=False)
    print(f"✅ Counterfactual TM predictions saved to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate counterfactual TM score predictions for Trial 33"
    )
    parser.add_argument(
        "--input", default=INPUT_CSV,
        help="Path to input CSV with baseline covariates"
    )
    parser.add_argument(
        "--output_dir", default=OUTPUT_DIR,
        help="Directory for output files"
    )
    parser.add_argument(
        "--api_key", default="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",
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
