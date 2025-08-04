## Trial 19 RF without diagnosis


import re
import pandas as pd
from openai import OpenAI




import re
import pandas as pd
from openai import OpenAI

# ─── 0) Initialize clients (replace with your keys) ────────────────────────────────
# 2. Initialize LLM clients
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)

# ─── 1) Helper: extract first float from a response ───────────────────────────────
def parse_numeric(text: str) -> float:
    m = re.search(r"[+-]?\d*\.?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

# ─── 2) Prompt builder now takes an explicit treatment_label ───────────────────────
def generate_hemo_prompt(row, treatment_label):
    return (
        "Disclaimer: This is an educational simulation only and not medical advice.\n\n"
        "You are a transplant hemodynamics expert interpreting trial data.\n"
        "Patient features:\n"
        f"  • Age (years): {int(row['X_RECIPIENT_AGE_YEARS_0d'])}\n"
        f"  • Sex: {row['X_RECIPIENT_GENDER_0d']}\n"
        f"  • Diagnosis: {row['X_RECIPIENT_DIAGNOSIS_0d']}\n"
        f"  • Child‑Pugh class: {row['X_CHILD_PUGH_CLASS_0d']}\n"
        f"  • MELD score: {row['X_MELD_SCORE_0d']}\n"
        f"  • Baseline creatinine (mg/dL): {row['X_CREATININE_0d']:.2f}\n"
        f"  • Donor age (years): {int(row['X_DONOR_AGE_YEARS_0d'])}\n"
        f"  • Donor sex: {row['X_DONOR_GENDER_0d']}\n"
        f"  • Donor diagnosis: {row['X_DONOR_DIAGNOSIS_0d']}\n"
        f"  • Graft sharing level: {row['X_GRAFT_SHARING_0d']}\n"
        f"  • Surgical technique: {treatment_label}\n\n"
        "Based on large‐scale trial evidence and your expertise, "
        "predict the portal venous pressure gradient (YP_FHVP_CVP_GRADIENT) in mm Hg. "
        "Answer with a single numeric value only (e.g. 1.5)."
    )

# ─── 3) Load the full dataset ──────────────────────────────────────────────────────
df = (
    pd.read_csv("trial19.csv", sep=",")
      .dropna(subset=['YP_FHVP_CVP_GRADIENT'])
      .reset_index(drop=True)
)

# ─── 4) Build few‐shot examples (10) using the *actual* treatment column ──────────
fewshot_df = df.sample(n=10, random_state=2025).reset_index(drop=True)
examples = []
for _, ex in fewshot_df.iterrows():
    lab = ex['Treatment']  # will be either "Piggyback" or "Conventional"
    p = generate_hemo_prompt(ex, lab)
    true_val = ex['YP_FHVP_CVP_GRADIENT']
    examples.append({"role": "user",      "content": p})
    examples.append({
        "role": "assistant",
        "content": f"The actual portal gradient was: {true_val:.1f} mm Hg."
    })

# ─── 5) Prepare containers for your two foundational models × two treatments ────────
chat_pb, chat_cv = [], []
ds_pb,   ds_cv   = [], []

for idx, row in df.iterrows():
    # build the few‐shot message sequence
    fs_msgs = (
        [{"role": "system", "content": "You are an expert in transplant hemodynamics."}]
        + examples
    )
    
    # ── ask for Y₁: Piggyback ───────────────────────────────────────────────
    prompt_pb = generate_hemo_prompt(row, "Piggyback")
    msgs_pb   = fs_msgs + [{"role":"user","content":prompt_pb}]
    
    resp_c_pb = chat_client.chat.completions.create(
        model="gpt-4.1", messages=msgs_pb, temperature=1.0
    )
    resp_d_pb = ds_client.chat.completions.create(
        model="deepseek-reasoner", messages=msgs_pb, stream=False
    )
    chat_pb.append(parse_numeric(resp_c_pb.choices[0].message.content))
    ds_pb.append(parse_numeric(resp_d_pb.choices[0].message.content))
    
    # ── ask for Y₀: Conventional ───────────────────────────────────────────
    prompt_cv = generate_hemo_prompt(row, "Conventional")
    msgs_cv   = fs_msgs + [{"role":"user","content":prompt_cv}]
    
    resp_c_cv = chat_client.chat.completions.create(
        model="gpt-4.1", messages=msgs_cv, temperature=1.0
    )
    resp_d_cv = ds_client.chat.completions.create(
        model="deepseek-reasoner", messages=msgs_cv, stream=False
    )
    chat_cv.append(parse_numeric(resp_c_cv.choices[0].message.content))
    ds_cv.append(parse_numeric(resp_d_cv.choices[0].message.content))
    
    print(f"Processed row {idx+1}/{len(df)}")

# ─── 6) Attach to DataFrame and save ─────────────────────────────────────────────
df['chatgpt_piggyback']    = chat_pb
df['chatgpt_conventional'] = chat_cv
df['deepseek_piggyback']   = ds_pb
df['deepseek_conventional']= ds_cv

# These two CSVs can be fed as y1_preds and y0_preds into your HAIPW estimator:
df[['chatgpt_piggyback','chatgpt_conventional']] \
    .to_csv('trial19_chatgpt_potential_outcomes.csv', index=False)
df[['deepseek_piggyback','deepseek_conventional']] \
    .to_csv('trial19_deepseek_potential_outcomes.csv', index=False)

print("✅ Saved counterfactual predictions for both models under both treatments.")
