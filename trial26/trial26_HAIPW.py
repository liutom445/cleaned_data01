


import re
import pandas as pd
from openai import OpenAI

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_numeric(text: str) -> float:
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_adherence_prompt(row, treatment_label):
    """
    Build prompt asking for the expected 6‑month change in adherence
    under the specified treatment for this patient.
    """
    return (
        "Disclaimer: This is an educational simulation only.\n\n"
        "You are an expert in HIV adherence interventions.\n"
        "Patient baseline data:\n"
        f"  • Age category: {row['X_Agecat_0m']}\n"
        f"  • Education level: {row['X_Education_0m']}\n"
        f"  • Ethnicity: {row['X_Ethnicity_0m']}\n"
        f"  • Social support score: {row['X_Socialsupport_0m']}\n"
        f"  • Baseline TB status: {row['X_TB_status_0m']}\n"
        f"  • Baseline OI index: {row['X_OI_index_0m']}\n"
        f"  • Baseline weight: {row['X_weight_0m']} kg\n"
        f"  • Baseline CD4 count: {row['X_CD4_0m']}\n"
        f"  • Baseline viral load: {row['X_viral_load_0m']}\n"
        f"  • Baseline adherence: {row['X_Adherence_0m']}\n\n"
        f"Assigned intervention: {treatment_label}.\n"
        "Based on your expertise, predict the change in adherence score\n"
        "from baseline to 6 months. Answer with a single numeric value."
    )


# ─── 1) Load trial‑26 data ──────────────────────────────────────────────────────
df = pd.read_csv("trial26.csv")

# ─── 2) Build few‑shot examples (optional) ─────────────────────────────────────
# Here we skip few‑shot; you can sample from trial26 itself if desired:
examples = []

# ─── 3) Initialize LLM clients ─────────────────────────────────────────────────
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)

# ─── 4) Predict counterfactuals ────────────────────────────────────────────────
chat_y1, chat_y0 = [], []
ds_y1,   ds_y0   = [], []

for _, row in df.iterrows():
    # Build shared context
    base_msgs = [{"role":"system","content":"You are an HIV adherence expert."}] + examples

    # --- Under treatment (arm=1) ---
    msg1 = base_msgs + [{
    "role": "user",
    "content": generate_adherence_prompt(row, "Reminder module")
    }]
    r1_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msg1)
    r1_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msg1, stream=False)
    chat_y1.append(parse_numeric(r1_c.choices[0].message.content))
    ds_y1.append(parse_numeric(r1_d.choices[0].message.content))

    # --- Under control (arm=0) ---
    msg0 = base_msgs + [{
    "role": "user",
    "content": generate_adherence_prompt(row, "Standard care")
    }]
    r0_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msg0)
    r0_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msg0, stream=False)
    chat_y0.append(parse_numeric(r0_c.choices[0].message.content))
    ds_y0.append(parse_numeric(r0_d.choices[0].message.content))

# ─── 5) Save to CSV ───────────────────────────────────────────────────────────
df["chatgpt_Δadh_6m_1"] = chat_y1
df["chatgpt_Δadh_6m_0"] = chat_y0
df["deepseek_Δadh_6m_1"] = ds_y1
df["deepseek_Δadh_6m_0"] = ds_y0

# These files feed directly as y1_preds / y0_preds into your HAIPW routine:
df[["chatgpt_Δadh_6m_1","chatgpt_Δadh_6m_0"]] \
   .to_csv("trial26_chatgpt_potential_adherence.csv", index=False)
df[["deepseek_Δadh_6m_1","deepseek_Δadh_6m_0"]] \
   .to_csv("trial26_deepseek_potential_adherence.csv", index=False)

print("✅ Counterfactual adherence predictions saved for trial‑26.")
