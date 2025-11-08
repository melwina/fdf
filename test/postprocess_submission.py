import pandas as pd
import ast
import sys

# Get command line arguments with defaults
input_csv = sys.argv[1] 
output_txt = sys.argv[2] 
quantile = float(sys.argv[3]) 

df = pd.read_csv(input_csv)

df['pred_first'] = df['pred'].apply(lambda x: ast.literal_eval(x)[0])
df['pred_first'] = pd.to_numeric(df['pred_first'], errors='coerce')

group_median = (
    df.groupby(['Skin Tone', 'Predicted Gender'])['pred_first']
      .quantile(quantile)
      .reset_index()
      .rename(columns={'pred_first': 'median_pred_first'})
)

df = df.merge(group_median, on=['Skin Tone', 'Predicted Gender'], how='left')
df['out'] = (df['pred_first'] <= df['median_pred_first']).astype(int)

df[['Image Path', 'out']].to_csv(output_txt, sep='\t', index=False, header=False)