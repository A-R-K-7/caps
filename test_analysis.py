import pandas as pd
import sys

df = pd.read_csv('research/results/research_2887d0b3_episodes.csv')
test_df = df[df['split'] == 'test']

print('### Aggregate Results (Test Split)\n')
print('| Method | Budget | Success Rate (%) | Mean Tokens | Precision | Recall |')
print('|--------|--------|------------------|-------------|-----------|--------|')

grouped = test_df.groupby(['method', 'budget_frac']).agg({
    'success': 'mean',
    'total_tokens': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'task_id': 'count'
}).reset_index()

for _, row in grouped.iterrows():
    print(f"| {row['method']:<20} | {row['budget_frac']:.2f} | {row['success']*100:>5.1f}% | {row['total_tokens']:>4.0f} | {row['precision']:.3f} | {row['recall']:.3f} |")
