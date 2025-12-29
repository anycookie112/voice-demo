import pandas as pd
df = pd.read_csv("/home/robust/voice-demo-new/data_visualisation/playground/sales_data_sample.csv", encoding="latin-1")

# Aggregate total sales by year and quarter
agg = df.groupby(['YEAR_ID', 'QTR_ID'], as_index=False)['SALES'].sum()

# Create x-axis labels like "2020 Q1"
agg['label'] = agg['YEAR_ID'].astype(str) + " Q" + agg['QTR_ID'].astype(str)

# Prepare chart data
chart_data = {
    "x": agg['label'].tolist(),
    "y": agg['SALES'].tolist(),
    "x_name": "Quarter",
    "y_name": "Total Sales"
}

import json
print(json.dumps(chart_data))