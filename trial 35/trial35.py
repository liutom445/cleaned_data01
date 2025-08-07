import os
import re
import time
import pandas as pd
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV            = "trial 35/trial35.csv"
OUTPUT_UNCOND_CSV    = "trial 35/trial35_unconditional_predictions.csv"
OUTPUT_CF_CSV        = "trial 35/trial35_counterfactual_predictions.csv"
CHAT_MODEL           = "gpt-4.1"
DEESEEK_MODEL        = "deepseek-chat"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """
    Extract the first numeric value from an LLM response.
    Treat 'nan' (case-insensitive) as NaN.
    """
    txt = text.strip()
    if txt.lower() == "nan":
        return float("nan")
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", txt)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_uncond_prompt(row: pd.Series, treatment_label: str) -> str:
    """
    Prompt to predict total weight gain from baseline to delivery
    under a given intervention.
    """
    lines = [
        "Disclaimer: This is an educational simulation only.",
        "",
        "You are an expert in maternal health outcomes and weight gain.",
        "",
        "Patient baseline data:",
        f"  • Pre‐pregnancy BMI: {row['X_preg_BMI_0w']:.1f} kg/m²",
        f"  • BMI at week 0 of pregnancy: {row['X_BMI_0w']:.1f} kg/m²",
        f"  • Psychological well‐being index (X_PGWB_index_0w): {row['X_PGWB_index_0w']}",
        f"  • Anxiety score (X_Anxiety_0w): {row['X_Anxiety_0w']}",
        f"  • Depressed mood score (X_Depressed_0w): {row['X_Depressed_0w']}",
        f"  • Positive well‐being score (X_Pos_Wellbeing_0w): {row['X_Pos_Wellbeing_0w']}",
        f"  • Self‐control score (X_Self_control_0w): {row['X_Self_control_0w']}",
        f"  • General health score (X_General_Health_0w): {row['X_General_Health_0w']}",
        f"  • Vitality score (X_Vitality_0w): {row['X_Vitality_0w']}",
        "",
        f"Intervention assigned: **{treatment_label}**.",
        "",
        "Predict the total weight gain (from baseline to delivery) in kilograms.",
        "Reply with only a single numeric value (e.g., 12.3)."
    ]
    return "\n".join(lines)


def generate_cf_prompt(row: pd.Series) -> str:
    """
    Prompt to predict the counterfactual total weight gain under the
    unassigned intervention, using the true observed gain.
    """
    actual_arm = row["Treatment"]           # "Exercise" or "Control"
    other_arm  = "Control" if actual_arm == "Exercise" else "Exercise"
    actual_gain = row["YP_delta_Total_weight_gain_delivery"]

    lines = [
        "Disclaimer: This is an educational simulation only.",
        "",
        "You are an expert in maternal health outcomes and weight gain.",
        "",
        "Patient baseline data:",
        f"  • Pre‐pregnancy BMI: {row['X_preg_BMI_0w']:.1f} kg/m²",
        f"  • BMI at week 0 of pregnancy: {row['X_BMI_0w']:.1f} kg/m²",
        f"  • Psychological well‐being index (X_PGWB_index_0w): {row['X_PGWB_index_0w']}",
        f"  • Anxiety score (X_Anxiety_0w): {row['X_Anxiety_0w']}",
        f"  • Depressed mood score (X_Depressed_0w): {row['X_Depressed_0w']}",
        f"  • Positive well‐being score (X_Pos_Wellbeing_0w): {row['X_Pos_Wellbeing_0w']}",
        f"  • Self‐control score (X_Self_control_0w): {row['X_Self_control_0w']}",
        f"  • General health score (X_General_Health_0w): {row['X_General_Health_0w']}",
        f"  • Vitality score (X_Vitality_0w): {row['X_Vitality_0w']}",
        "",
        f"The patient as assigned to **{actual_arm}** had a total weight gain of **{actual_gain:.2f} kg**.",
        f"What would their total weight gain have been if they instead received **{other_arm}**?",
        "",
        "Reply with only the numeric gain in kilograms (e.g., 10.5)."
    ]
    return "\n".join(lines)


def main():
    # 1) Load & filter to the two arms
    df = (
        pd.read_csv(INPUT_CSV)
          .dropna(subset=["Treatment", "YP_delta_Total_weight_gain_delivery"])
          .query("Treatment in ['Exercise','Control']")
          .reset_index(drop=True)
    )

    # 2) Initialize LLM clients
    chat_client = OpenAI(api_key="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")
    ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


    # 3) Prepare containers
    chat_y1, chat_y0 = [], []
    ds_y1,   ds_y0   = [], []
    cf_preds         = []
    total            = len(df)
    start            = time.time()

    print(f"Starting {total} patients — unconditional + counterfactual queries...")

    # 4) Loop over patients
    for i, row in df.iterrows():
        # Unconditional Y(1): Exercise
        p1 = generate_uncond_prompt(row, "Exercise")
        r1_c = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":"You are an expert in maternal health outcomes."},
                {"role":"user","content":p1}
            ]
        )
        r1_d = ds_client.chat.completions.create(
            model=DEESEEK_MODEL,
            messages=[
                {"role":"system","content":"You are an expert in maternal health outcomes."},
                {"role":"user","content":p1}
            ],
            stream=False
        )
        y1_c = parse_numeric(r1_c.choices[0].message.content)
        y1_d = parse_numeric(r1_d.choices[0].message.content)
        chat_y1.append(y1_c)
        ds_y1.append(y1_d)

        # Unconditional Y(0): Control
        p0 = generate_uncond_prompt(row, "Control")
        r0_c = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":"You are an expert in maternal health outcomes."},
                {"role":"user","content":p0}
            ]
        )
        r0_d = ds_client.chat.completions.create(
            model=DEESEEK_MODEL,
            messages=[
                {"role":"system","content":"You are an expert in maternal health outcomes."},
                {"role":"user","content":p0}
            ],
            stream=False
        )
        y0_c = parse_numeric(r0_c.choices[0].message.content)
        y0_d = parse_numeric(r0_d.choices[0].message.content)
        chat_y0.append(y0_c)
        ds_y0.append(y0_d)

        # Counterfactual for unassigned arm
        cf_prompt = generate_cf_prompt(row)
        cf_resp   = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":"You are an expert in maternal health outcomes."},
                {"role":"user","content":cf_prompt}
            ]
        )
        try:
            cf_val = parse_numeric(cf_resp.choices[0].message.content)
        except ValueError:
            print(f"⚠️ Row {i} non-numeric CF response; setting NaN.")
            cf_val = float("nan")
        cf_preds.append(cf_val)

        # Progress & ETA every 30 or last
        if (i+1) % 30 == 0 or (i+1) == total:
            elapsed = time.time() - start
            eta     = (elapsed / (i+1)) * (total - (i+1))
            print(f"🛎️ {i+1}/{total} | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    # 5) Save unconditional potential outcomes
    df["chatgpt_y1"], df["chatgpt_y0"] = chat_y1, chat_y0
    df["deepseek_y1"], df["deepseek_y0"] = ds_y1, ds_y0
    df[["chatgpt_y1","chatgpt_y0"]].to_csv(OUTPUT_UNCOND_CSV, index=False)
    df[["deepseek_y1","deepseek_y0"]].to_csv("trial35_deepseek_potential_outcomes.csv", index=False)

    # 6) Save counterfactuals
    df["chatgpt_cf_weight_gain"] = cf_preds
    df[["chatgpt_cf_weight_gain"]].to_csv(OUTPUT_CF_CSV, index=False)

    print("✅ Saved unconditional and counterfactual predictions for Trial 35.")

if __name__ == "__main__":
    main()
