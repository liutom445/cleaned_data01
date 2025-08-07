import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = "trial 33/trial33.csv"
OUTPUT_DIR = "trial 33/"
MODEL      = "gpt-4.1"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response, or NaN if 'nan'."""
    txt = text.strip()
    if txt.lower() == "nan":
        return float("nan")
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", txt)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_cf_prompt(row: pd.Series) -> str:
    """
    Build a counterfactual prompt for Trial 33:
    - Lists baseline covariates
    - States the actual 6-month Therapeutic Misconception score under assigned arm
    - Asks what it would have been under the other arm
    """
    actual_arm = row["Treatment"]  # "Reframing" or "Control"
    other_arm  = "Control" if actual_arm == "Reframing" else "Reframing"
    actual_y   = row["YP_TM_total_score"]

    lines = [
        "Disclaimer: This is an educational simulation only, not clinical advice.",
        "",
        "You are an expert in clinical trial outcome prediction for Therapeutic Misconception.",
        "",
        "Patient baseline covariates:",
        f"  • Treatment arm: {actual_arm}",
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
        f"The patient as assigned to **{actual_arm}** had a 6-month Therapeutic Misconception total score of **{actual_y:.2f}**.",
        f"What would their 6-month score have been if they instead received **{other_arm}**?",
        "",
        "Please reply with only the numeric score (e.g., 3.5)."
    ]
    return "\n".join(lines)


def main():
    # 1) Load & filter to the two arms
    df = pd.read_csv(INPUT_CSV)

    # 2) Initialize OpenAI client
    client = OpenAI(api_key="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")


    total       = len(df)
    cf_preds    = []
    query_count = 0
    start_time  = time.time()

    print(f"Starting {total} counterfactual queries (ChatGPT {MODEL})...")

    # 3) Loop over each patient
    for idx, row in df.iterrows():
        system_msg  = {
            "role": "system",
            "content": "You are an expert in clinical trial outcome prediction for Therapeutic Misconception."
        }
        user_prompt = generate_cf_prompt(row)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, {"role": "user", "content": user_prompt}]
        )
        content = resp.choices[0].message.content

        # 4) Parse numeric, catch any non-numeric responses
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

    # 6) Save only the counterfactual column
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"trial33_{MODEL}_cf.csv")
    df["chatgpt_cf_TMscore_6m"] = cf_preds
    df[["chatgpt_cf_TMscore_6m"]].to_csv(output_path, index=False)

    print(f"✅ Counterfactual predictions saved to {output_path}")


if __name__ == "__main__":
    main()
