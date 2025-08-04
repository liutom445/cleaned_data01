import os
import re
import pandas as pd
import numpy as np
from openai import OpenAI

def parse_numeric(text: str) -> float:
    """
    Find the first float-like token in text and return it.
    Raises ValueError if none is found.
    """
    m = re.search(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))


# 1. Load Trial 4 data & encode treatment
df4 = pd.read_csv("/Users/hongyiliu/Desktop/Research/collection/SS 25/Meeting 0729/cleaned_data/trial4/trial4.csv").dropna(subset=["Treatment", "YP_delta_P3NP_6w"])
df4['W'] = (df4['Treatment'] == 'VD').astype(int)

# 2. Identify covariate columns
covariates = [c for c in df4.columns if c not in ("Treatment", "YP_delta_P3NP_6w", "W")]

# 3. Initialize OpenAI and DeepSeek clients
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# 4. Prompt builder that injects covariate names, values and definitions
def generate_query(row, treatment_label):
    """
    Build a clinical prompt including all covariates with their values
    and definitions, for the counterfactual scenario with treatment_label.
    """
    # covariate definitions per Sriphoosanaphan et al.
    defs = {
        'X_FIB4_0w':   'Fibrosis-4 score at baseline, non-invasive fibrosis index',
        'X_APRI_0w':   'AST-to-platelet ratio index at baseline, another fibrosis marker',
        'X_VD_0w':     'Serum 25-hydroxyvitamin D at baseline (ng/mL)',
        'X_AST_0w':    'Aspartate aminotransferase at baseline (U/L)',
        'X_ALT_0w':    'Alanine aminotransferase at baseline (U/L)',
        'X_Plt_0w':    'Platelet count at baseline (10^3 cells/μL)',
        'X_TGF_0w':    'TGF-β1 at baseline, pro-fibrogenic cytokine (ng/mL)',
        'X_TIMP_0w':   'TIMP-1 at baseline, matrix breakdown inhibitor (ng/mL)',
        'X_MMP_0w':    'MMP-9 at baseline, fibrolytic enzyme (ng/mL)',
        'X_P3NP_0w':   'P3NP at baseline, collagen-degradation marker (ng/mL)'
    }
    # build feature lines
    feature_lines = []
    for cov in covariates:
        val = row[cov]
        desc = defs.get(cov, '')
        feature_lines.append(f"  • {cov} = {val} ({desc})")

    disclaimer = (
        "Disclaimer: This simulation is for educational and research purposes only "
        "and does not constitute medical advice.\n\n"
    )
    if treatment_label.lower() == 'placebo':
        treat_text = 'They were randomized to receive placebo.'
    else:
        treat_text = f'You decide to treat them with {treatment_label}.'

    prompt = (
        disclaimer
        + "Patient baseline characteristics (covariate = value):\n"
        + "\n".join(feature_lines)
        + "\n\n" + treat_text + "\n\n"
        + "Based on your expertise, what is the expected change in P3NP at 6 weeks? "
        + "Respond with a single real number (e.g., -0.5 or 2.3)."
    )
    return prompt

# 5. Build few‑shot examples using a small sample
fewshot_df = df4.sample(n=5, random_state=2025)
examples = []
for _, row in fewshot_df.iterrows():
    label = 'Vitamin D2' if row['W'] == 1 else 'Placebo'
    prompt_ex = generate_query(row, label)
    true_lbl  = f"The actual ΔP3NP is: {row['YP_delta_P3NP_6w']:.2f}"
    examples.append({"role": "user",      "content": prompt_ex})
    examples.append({"role": "assistant", "content": true_lbl})

# 6. Prepare storage for counterfactual predictions
chat_cf_treat   = []
chat_cf_control = []
ds_cf_treat     = []
ds_cf_control   = []

# 7. Loop through patients, request counterfactuals, notify every 20
for idx, row in df4.iterrows():
    system_msg = {"role": "system", "content": "You are a hepatology simulation engine."}
    # treated scenario
    msgs_t = [system_msg] + examples + [{"role": "user", "content": generate_query(row, 'Vitamin D2')}]
    resp_t = chat_client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=msgs_t,         # adds a bit of randomness
    top_p=0.9,                # nucleus sampling
    frequency_penalty=0.8,    # discourages repeated tokens
    presence_penalty=0.5      # discourages repeating topics
)

    chat_cf_treat.append(parse_numeric(resp_t.choices[0].message.content))
    resp_dt = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs_t, stream=False)
    ds_cf_treat.append(parse_numeric(resp_dt.choices[0].message.content))

    # placebo scenario
    msgs_c = [system_msg] + examples + [{"role": "user", "content": generate_query(row, 'Placebo')}]
    resp_c = chat_client.chat.completions.create(model="gpt-4.1-mini", messages=msgs_c)
    chat_cf_control.append(parse_numeric(resp_c.choices[0].message.content))
    resp_dc = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs_c, stream=False)
    ds_cf_control.append(parse_numeric(resp_dc.choices[0].message.content))

    # notify every 20
    if (idx + 1) % 25 == 0:
        print(f"🔔 Processed {idx+1} / {len(df4)} patients")

# 8. Attach and save
output_cols = ["Treatment", "W", "YP_delta_P3NP_6w"] + covariates + [
    "chatgpt_cf_VitD2", "chatgpt_cf_Placebo",
    "deepseek_cf_VitD2", "deepseek_cf_Placebo"
]

df4['chatgpt_cf_VitD2']    = chat_cf_treat
df4['chatgpt_cf_Placebo']  = chat_cf_control
df4['deepseek_cf_VitD2']   = ds_cf_treat
df4['deepseek_cf_Placebo'] = ds_cf_control

df4[output_cols].to_csv("/Users/hongyiliu/Desktop/Research/collection/SS 25/Meeting 0729/cleaned_data/trial4/trial4_cf.csv", index=False)
print("✅ Saved Trial 4 dataset with covariates, definitions, and counterfactual predictions.")
