import pandas as pd
import numpy as np

WEIGHTS = {
    'Profit': 0.125,
    'Expected_Payoff': 0.125,
    'Profit_Factor': 0.25,
    'Recovery_Factor': 0.25,
    'Sharpe_Ratio': 0.25,
    'Max Equity_DD_%': -0.25
}

df = pd.read_csv("result/rsi_results.csv")

# Remove duplicate records
print(f"Before removing duplicates: {len(df)} rows")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


scores = []
for metric, weight in WEIGHTS.items():
    if weight >= 0:
        scores.append(weight * normalize(df[metric]))
    else:
        scores.append(weight * (1 - normalize(df[metric])))

df['Score'] = np.sum(scores, axis=0)
df['Score'] = df['Score'].round(3)
df_sorted = df.sort_values('Score', ascending=False)

# Apply filters: Score > 0.1, Profit > 1000, Avg_Trades_Per_Week > 1
df_filtered = df_sorted[
    (df_sorted['Score'] > 0.1) & 
    (df_sorted['Profit'] > 1000) & 
    (df_sorted['Avg_Trades_Per_Week'] > 1)
]

print(f"Original data: {len(df_sorted)} rows")
print(f"After filtering: {len(df_filtered)} rows")

df_filtered.to_csv("result/rsi_results_10_filtered.csv", index=False)
