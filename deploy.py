import json
import subprocess
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

import numpy as np

with open('docs/metrics.json', 'r') as f:
    metrics = json.load(f)

accuracy = round(metrics['accuracy'], 2)
auc = round(metrics['auc'], 2)
precision = round(metrics['precision'], 2)
recall = round(metrics['recall'], 2)
f1_score = round(metrics['f1_score'], 2)


# Load the data from the URL
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
data = pd.read_excel(url, header=1)

# Create the charts
chart1 = px.scatter(data, x='BILL_AMT1', y='PAY_AMT1', color='SEX', title='Scatter Plot of Bill Amount vs. Payment Amount', 
                    width=500, height=500, color_discrete_sequence=['#0096C7', '#D90429'])
chart2 = px.histogram(data, x='EDUCATION', title='Distribution of Education Level', width=500, height=500, 
                      color_discrete_sequence=['#FFC300'])
chart3 = px.histogram(data, x='MARRIAGE', title='Distribution of Marital Status', width=500, height=500, 
                      color_discrete_sequence=['#83AF9B'])
chart4 = px.bar(data, x='SEX', y='LIMIT_BAL', title='Credit Limits by Gender', width=500, height=500, 
                color_discrete_sequence=['#D90429'])

# Arrange the charts in a 2x2 grid
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=('Scatter Plot of Bill Amount vs. Payment Amount', 
                                                       'Distribution of Education Level', 
                                                       'Distribution of Marital Status', 
                                                       'Credit Limits by Gender'))
fig.add_trace(chart1.data[0], row=1, col=1)
fig.add_trace(chart2.data[0], row=1, col=2)
fig.add_trace(chart3.data[0], row=2, col=1)
fig.add_trace(chart4.data[0], row=2, col=2)
fig.update_layout(width=800, height=800)

# Create a beautiful table for the metrics
metrics_df = pd.DataFrame({'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score'], 
                           'Value': [accuracy, auc, precision, recall, f1_score]})

metrics_table = go.Figure(data=[go.Table(
    header=dict(values=list(metrics_df.columns),
                fill_color='#0096C7',
                font=dict(color='white', size=12),
                align='center'),
    cells=dict(values=[metrics_df.Metric, metrics_df.Value],
               fill_color='#F0F8FF',
               font=dict(color='black', size=12),
               align='center'))
])
metrics_table.update_layout(title='Model Metrics', width=700, height=400) # adjust width and height here

# Save the grid and the metrics to an HTML file
with open('docs/index.html', 'w') as f:
    # Write the HTML file header
    f.write('<html><head><title>Logistic regression</title></head><body>')
    # Write the charts and metrics to the HTML file
    f.write('<table style="width:100%"><tr><td>')
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write('</td><td>')
    f.write(metrics_table.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write('</td></tr><tr><td>')
    # Skip the confusion matrix
    f.write('</td></tr></table></body></html>')



# Post a message to the user indicating that the deployment was successful
message = f"🚀 @{os.environ['GITHUB_ACTOR']}, your model has been deployed successfully! 🎉🎉🎉 <br>See, https://turgut090.github.io/mlops/<br>"
issue_number = os.environ['ISSUE_NUMBER']
uri = os.environ['URI']
auth_header = f"Authorization: token {os.environ['GITHUB_TOKEN']}"
content_type_header = "Content-Type: application/json"
data = {"body": message}
data_str = json.dumps(data)
cmd = f"curl -X POST -sSL -d '{data_str}' -H '{auth_header}' -H '{content_type_header}' {uri}/repos/{os.environ['GITHUB_REPOSITORY']}/issues/{issue_number}/comments"
subprocess.check_call(cmd, shell=True)



