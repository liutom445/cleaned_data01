
import re
import pandas as pd
from openai import OpenAI

def parse_numeric(text: str) -> float:
    m = re.search(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', text)
    if not m:
        raise ValueError(f"No numeric value found in response: {text!r}")
    return float(m.group(0))

def generate_query(treatment_label, age, sex, comorbidities, severity):
    disclaimer = (
        "Disclaimer: This simulation is for educational and research purposes only "
        "and does not constitute medical advice.\n\n"
    )
    comb_text = (
        "no known comorbidities"
        if pd.isna(comorbidities) or comorbidities.lower() == "not known"
        else comorbidities
    )
    if treatment_label == "placebo":
        treat_text = "They were randomized to receive a placebo."
    else:
        treat_text = f"You decide to treat them with {treatment_label}."

    return (
        disclaimer +
        f"Suppose you’re an expert in COVID-19 intervention. "
        f"You have recently admitted a {int(age)}-year-old {sex.lower()} patient "
        f"with {severity.lower()} COVID-19. They have {comb_text}. "
        f"{treat_text} "
        "Based on your clinical experience, what is the probability this patient tests negative by day 6? "
        "Respond with only a real number in (0,1)."
    )

# 1. Load data
df = pd.read_csv("trial8_df8_all.csv")

# 2. Initialize LLM clients
chat_client = OpenAI(api_key="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")
ds_client   = OpenAI(
    api_key="sk-873cf9e994684da992b866873324946b",
    base_url="https://api.deepseek.com"
)


# 3. Build few‑shot examples (now mapping W→labels)
fewshot_df = df.sample(n=10, random_state=2025).reset_index(drop=True)
examples = []
for _, row in fewshot_df.iterrows():
    label = "Ivermectin" if row['W'] == 1 else "placebo"
    ex_prompt = generate_query(
        label,
        row['X_Age_0d'], row['X_Sex_0d'],
        row['X_comorbidities_0d'], row['X_severity_admission_0d']
    )
    examples.append({"role": "user", "content": ex_prompt})
    examples.append({"role": "assistant", "content": f"{int(row['Y'])}"})

# 4. Loop to get both Y₁ and Y₀ for each LLM
chat_y1, chat_y0 = [], []
ds_y1,   ds_y0   = [], []

for _, row in df.iterrows():
    age, sex, comb, sev = (
        row['X_Age_0d'], row['X_Sex_0d'],
        row['X_comorbidities_0d'], row['X_severity_admission_0d']
    )

    # base messages
    base_msgs = [{"role": "system", "content": "You are a biomedical simulation engine."}] + examples

    # --- Y₁(i): “Ivermectin” ---
    msgs1 = base_msgs + [{"role": "user", "content": generate_query("Ivermectin", age, sex, comb, sev)}]
    r1_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msgs1)
    r1_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs1, stream=False)
    chat_y1.append(parse_numeric(r1_c.choices[0].message.content))
    ds_y1.append(parse_numeric(r1_d.choices[0].message.content))

    # --- Y₀(i): “placebo” ---
    msgs0 = base_msgs + [{"role": "user", "content": generate_query("placebo", age, sex, comb, sev)}]
    r0_c = chat_client.chat.completions.create(model="gpt-4.1", messages=msgs0)
    r0_d = ds_client.chat.completions.create(model="deepseek-chat", messages=msgs0, stream=False)
    chat_y0.append(parse_numeric(r0_c.choices[0].message.content))
    ds_y0.append(parse_numeric(r0_d.choices[0].message.content))

# 5. Save to CSV
df['chatgpt_y1'] = chat_y1
df['chatgpt_y0'] = chat_y0
df['deepseek_y1'] = ds_y1
df['deepseek_y0'] = ds_y0

df[['chatgpt_y1','chatgpt_y0']].to_csv('trial8_chatgpt_potential_outcomes.csv', index=False)
df[['deepseek_y1','deepseek_y0']].to_csv('trial8_deepseek_potential_outcomes.csv', index=False)

print("✅ Saved both Y0/Y1 predictions for ChatGPT and DeepSeek.")
