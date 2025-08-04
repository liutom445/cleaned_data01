
import re
import pandas as pd
from openai import OpenAI

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_recovery_prompt(row, treatment_label):
    """
    Build prompt asking for the expected time-to-recovery (in days)
    under the specified treatment for this patient.
    """
    return (
        "Disclaimer: This simulation is for educational and research purposes only.\n\n"
        "You are a clinical trial expert predicting recovery times.\n"
        "Patient baseline characteristics:\n"
        f"  • Age group: {row['X_agegrp_0d']}\n"
        f"  • Sex: {row['X_sex_0d']}\n"
        f"  • Fever at baseline: {row['X_fever_0d']}\n"
        f"  • Cough at baseline: {row['X_cough_0d']}\n"
        f"  • Respiratory difficulty: {row['X_respdiff_0d']}\n"
        f"  • Comorbidity present: {row['X_comorb_0d']}\n"
        f"  • Diabetes: {row['X_diabetes_0d']}\n"
        f"  • Hypertension: {row['X_hyperten_0d']}\n\n"
        f"Assigned treatment: {treatment_label}.\n"
        "Based on large-scale trial data and clinical expertise, "
        "predict this patient’s time to recovery (in days). "
        "Respond with a single numeric value."
    )

# ─── 1) Load Trial 2 data ──────────────────────────────────────────────────────
df = pd.read_csv("trial2.csv")

# ─── 2) Build few-shot examples from actual data ──────────────────────────────
fewshot_df = df.sample(n=10, random_state=2025).reset_index(drop=True)
examples = []
for _, ex in fewshot_df.iterrows():
    label = ex['Treatment']
    prompt = generate_recovery_prompt(ex, label)
    examples.append({"role": "user", "content": prompt})
    examples.append({"role": "assistant", "content": f"{float(ex['YP_recovery_time'])}"})

# ─── 3) Initialize LLM clients ─────────────────────────────────────────────────
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# ─── 4) Predict potential outcomes Y(1) and Y(0) ───────────────────────────────
chat_y1, chat_y0 = [], []
ds_y1,   ds_y0   = [], []

for _, row in df.iterrows():
    base_msgs = [{"role": "system", "content": "You are a clinical trial expert."}] + examples

    # Y1: Ivermectin+Doxycycline
    msgs1 = base_msgs + [{
        "role": "user",
        "content": generate_recovery_prompt(row, "Ivermectin+Doxycycline")
    }]
    r1_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msgs1)
    r1_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs1, stream=False)
    chat_y1.append(parse_numeric(r1_c.choices[0].message.content))
    ds_y1.append(parse_numeric(r1_d.choices[0].message.content))

    # Y0: Placebo
    msgs0 = base_msgs + [{
        "role": "user",
        "content": generate_recovery_prompt(row, "Placebo")
    }]
    r0_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msgs0)
    r0_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs0, stream=False)
    chat_y0.append(parse_numeric(r0_c.choices[0].message.content))
    ds_y0.append(parse_numeric(r0_d.choices[0].message.content))

# ─── 5) Save potential outcomes to CSV ─────────────────────────────────────────
df["chatgpt_y1"] = chat_y1
df["chatgpt_y0"] = chat_y0
df["deepseek_y1"] = ds_y1
df["deepseek_y0"] = ds_y0

df[["chatgpt_y1", "chatgpt_y0"]].to_csv("trial2_chatgpt_potential_outcomes.csv", index=False)
df[["deepseek_y1","deepseek_y0"]].to_csv("trial2_deepseek_potential_outcomes.csv", index=False)

print("✅ Counterfactual recovery-time predictions saved for Trial 2.")
