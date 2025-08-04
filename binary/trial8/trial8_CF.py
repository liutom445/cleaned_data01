import os
import re
import pandas as pd
import numpy as np
from openai import OpenAI

# Utility to extract numeric probabilities from LLM responses
def parse_numeric(text: str) -> float:
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

# 1. Load trial data & encode treatment indicator
df = (
    pd.read_csv("trial8_df8_all.csv")
)

df['Y_obs'] = df['Y']  # observed PCR-neg outcome

# 2. Initialize LLM clients (ChatGPT + DeepSeek)
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# 3. Counterfactual prompt builder
#    If W=1 (Ivermectin), ask: given outcome under Ivermectin, predict under placebo
#    If W=0 (Placebo), ask: given outcome under placebo, predict under Ivermectin
def generate_cf_query(treatment, age, sex, comorbidities, severity, y_obs):
    comb_text = (
        "no known comorbidities"
        if pd.isna(comorbidities) or comorbidities == "NA"
        else comorbidities
    )
    if treatment == 1:
        opposite = "placebo"
        known_txt = f"under Ivermectin, the patient tested negative by day 6"
    else:
        opposite = "Ivermectin"
        known_txt = f"under placebo, the patient tested negative by day 6"

    prompt = (
        "Disclaimer: This simulation is for educational and research purposes only and does not constitute medical advice.\n\n"
        "You are a biomedical simulation engine. "
        f"Suppose a {int(age)}-year-old {sex.lower()} patient with {comb_text} and {severity.lower()} COVID-19 was treated as follows: {known_txt}. "
        f"What is the probability this patient would test negative by day 6 if they had instead received {opposite}? "
        "Respond with only a real number between 0 and 1."
    )
    return prompt

# 4. Build few-shot examples (using factual outcomes as stand-ins)
fewshot_df = df.sample(n=5, random_state=42).reset_index(drop=True)
examples = []
for _, row in fewshot_df.iterrows():
    # create the sample prompt for user
    p_ex = generate_cf_query(
        row['W'], row['X_Age_0d'], row['X_Sex_0d'],
        row['X_comorbidities_0d'], row['X_severity_admission_0d'],
        row['Y_obs']
    )
    # use the factual probability as the demonstration answer
    a_ex = f"{row['Y_obs']:.2f}"
    examples.append({"role": "user", "content": p_ex})
    examples.append({"role": "assistant", "content": a_ex})

# 5. Run few-shot counterfactual predictions
cf_chat = []
cf_ds   = []
for _, row in df.iterrows():
    # generate prompt
    prompt = generate_cf_query(
        row['W'], row['X_Age_0d'], row['X_Sex_0d'],
        row['X_comorbidities_0d'], row['X_severity_admission_0d'],
        row['Y_obs']
    )
    # assemble few-shot conversation
    fs_msgs = [
        {"role": "system", "content": "You are a biomedical simulation engine."}
    ] + examples + [
        {"role": "user", "content": prompt}
    ]

    # ChatGPT few-shot
    resp_chat = chat_client.chat.completions.create(
        model="o3-mini",
        messages=fs_msgs,
        temperature=1.0
    )
    cf_chat.append(parse_numeric(resp_chat.choices[0].message.content))

    # DeepSeek few-shot
    resp_ds = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=fs_msgs,
        stream=False
    )
    cf_ds.append(parse_numeric(resp_ds.choices[0].message.content))

# 6. Attach counterfactual predictions and save
_df = df.copy()
_df['pred_cf_chat'] = cf_chat
_df['pred_cf_deepseek'] = cf_ds
_df.to_csv('trial8_cf.csv', index=False)

print("✅ Few-shot counterfactual predictions generated and saved.")
