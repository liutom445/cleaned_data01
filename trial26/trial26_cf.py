import os
import re
import pandas as pd
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_CSV  = "trial26/trial26.csv"       # Must include YP_delta_Adherence_6m and Treatment columns
OUTPUT_DIR = "trial26/"  # Directory to save results
CHAT_MODEL = "gpt-4.1"           # ChatGPT model for CF predictions

# ─── Helper Functions ─────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    """Extract the first numeric value from an LLM response."""
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


def generate_adherence_prompt(row,
                              treatment_label: str,
                              other_potential: float,
                              other_label: str) -> str:
    """
    Build prompt asking for the counterfactual 6-month change in adherence
    under the specified intervention, given the observed outcome from the other arm.
    """
    lines = [
        "Disclaimer: This is an educational simulation only.",
        "",
        "You are an expert in HIV adherence interventions.",
        "Patient baseline data:",
        f"  • Age category: {row['X_Agecat_0m']}",
        f"  • Education level: {row['X_Education_0m']}",
        f"  • Ethnicity: {row['X_Ethnicity_0m']}",
        f"  • Social support score: {row['X_Socialsupport_0m']}",
        f"  • Baseline TB status: {row['X_TB_status_0m']}",
        f"  • Baseline OI index: {row['X_OI_index_0m']}",
        f"  • Baseline weight: {row['X_weight_0m']} kg",
        f"  • Baseline CD4 count: {row['X_CD4_0m']}",
        f"  • Baseline viral load: {row['X_viral_load_0m']}",
        f"  • Baseline adherence: {row['X_Adherence_0m']}",
        "",
        f"The patient assigned to {other_label} has an observed 6-month adherence change of {other_potential}.",
        f"What would their 6-month adherence change be if they instead received {treatment_label}?",
        "",
        "Based on your expertise, predict the change in adherence score",
        "from baseline to 6 months. Answer with a single numeric value."
    ]
    return "\n".join(lines)

# ─── Load data and initialize ChatGPT client ──────────────────────────────────

df = pd.read_csv(INPUT_CSV)
chat_client = OpenAI(api_key="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")

# ─── Prepare result list ──────────────────────────────────────────────────────
cf_other = []  # counterfactual predictions for the unobserved arm

# ─── Iterate and collect CF predictions based on actual Y and Treatment ───────
for _, row in df.iterrows():
    system_msg = {"role": "system", "content": "You are an HIV adherence expert."}

    # Actual observed outcome and assignment
    actual_y = row['YP_delta_Adherence_6m']
    assigned = row['Treatment']  # values: "Standard care" or "Reminder module"

    if assigned == "Standard care":
        other_label     = "Standard care"
        treatment_label = "Reminder module"
    else:
        other_label     = "Reminder module"
        treatment_label = "Standard care"

    prompt = generate_adherence_prompt(
        row,
        treatment_label=treatment_label,
        other_potential=actual_y,
        other_label=other_label
    )
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system_msg, {"role": "user", "content": prompt}]
    )
    cf_other.append(parse_numeric(resp.choices[0].message.content))

# ─── Save only counterfactual predictions ─────────────────────────────────────

df['chatgpt_Dadh_6m_cf'] = cf_other
os.makedirs(OUTPUT_DIR, exist_ok=True)

df[['chatgpt_Dadh_6m_cf']] \
  .to_csv(os.path.join(OUTPUT_DIR, 'trial26_chatgpt_counterfactual_adherence.csv'), index=False)

print("✅ Counterfactual adherence predictions saved for Trial 26 (ChatGPT only).")
