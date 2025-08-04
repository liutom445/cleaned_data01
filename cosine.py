import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 1) Load & prepare
FILEPATH = "meta_data (2).xlsx"
df = pd.read_excel(FILEPATH).fillna("")
df["text_input"] = (
    df["Trial Number/Name"].astype(str)
    + " | Paper: " + df["Paper Name"].astype(str)
    + " | Cluster: "  + df["Cluster"].astype(str)
    + " | Outcome: "  + df["Primary Outcome"].astype(str)
    + " | Model: "    + df["Statistical Model"].astype(str)
    + " | Notes: "    + df["Text Data"].astype(str)
)

# 2) Fetch embeddings (openai>=1.0.0)
openai.api_key = "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"

response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=df["text_input"].tolist()
)

## transform unstructured covariates into numerical data 

# 3) Extract embeddings correctly
#    response.data is a list of OpenAIObject, each with .embedding
embeddings = np.vstack([item.embedding for item in response.data])

# 4) Compute cosine-similarity matrix
sim_matrix = cosine_similarity(embeddings, embeddings)

# 5) Build full neighbor table
records = []
k = 5
for i in range(len(df)):
    sims = sim_matrix[i].copy()
    sims[i] = -np.inf
    topk = np.argsort(sims)[-k:][::-1]
    for rank, j in enumerate(topk, start=1):
        records.append({
            "trial_idx":        i,
            "trial_name":       df.at[i, "Paper Name"],
            "neighbor_rank":    rank,
            "neighbor_idx":     j,
            "neighbor_name":    df.at[j, "Paper Name"],
            "similarity_score": sims[j]
        })

results_df = pd.DataFrame.from_records(records)

# 6) Save to CSV
output_path = "trial_nearest_neighbors.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved nearest‐neighbor results to {output_path}")


## MSE Table and Refinement of Writeup
## Counterfactual 

## Local Deployment of LLM. 

