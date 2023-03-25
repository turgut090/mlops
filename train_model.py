import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import joblib
import subprocess
import os
import json
import pandas as pd

# Get the command line argument for the data size multiplier and year-month filter
print('sys.argv[1]:', sys.argv[1])
print('sys.argv[2]:', sys.argv[2])
data_size_multiplier = int(sys.argv[1])
year_month_filter = sys.argv[2]


# Load the credit scoring dataset from the UCI Machine Learning Repository
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
df = pd.read_excel(url, header=1)
start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2021-12-01')
df['date'] = pd.date_range(start=start_date, end=end_date, periods=len(df))


# Randomly select rows based on the data_size_multiplier argument
sample_size = int(np.where(data_size_multiplier > len(df),len(df),data_size_multiplier))
df = df.sample(n=sample_size)

# Filter the dataframe by the year-month filter argument
df['year_month'] = df['date'].dt.to_period('M')
df = df[df['year_month'] == year_month_filter]
df.drop(columns=['year_month'], inplace=True)
df

# Make sure the default column is binary (0 or 1) and not categorical (1, 2, 3)
df['default payment next month'] = df['default payment next month'].apply(lambda x: 1 if x == 1 else 0)

X = df.drop(columns=['default payment next month','date','ID'])
y = df['default payment next month']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)


# if dir doesn't exist
dir_name = "trained"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Save the trained model to disk
joblib.dump(model, 'docs/model.pkl')

# Print the accuracy and AUC metrics
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_pred)

# Compute additional metrics
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Save the metrics to a file
metrics = {'accuracy': accuracy, 'auc': auc, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'confusion_matrix': cm.tolist()}
with open('docs/metrics.json', 'w') as f:
    json.dump(metrics, f)


# Prepare the reply message
reply_message = f"<b>Accuracy:</b> {accuracy:.3f} 🔥<br><b>AUC:</b> {auc:.3f} 👍<br>"
reply_message += f"<b>Precision:</b> {precision:.3f} 💪<br><b>Recall:</b> {recall:.3f} 🚀<br>"
reply_message += f"<b>F1-score:</b> {f1_score:.3f} 🌟<br><br>"
reply_message += f"<b>Confusion matrix:</b><br>{cm}<br>"
reply_message += f"<br><b>Sample size: {sample_size}<b> 📊<br>"
reply_message += f"<b>Sample period: {year_month_filter} 📊<b>"

# Add motivational messages based on the metrics
if accuracy > 0.8 and auc > 0.8 and f1_score > 0.8:
  reply_message += "<b>Congratulations!</b> 🎉🎉 Your model performed exceptionally well! 🚀🚀🚀"
elif accuracy > 0.7 and auc > 0.7 and f1_score > 0.7:
  reply_message += "<br>Your model performed well. 👏👏 Keep up the good work! 💪💪<br>"
else:
  reply_message += "<br>Your model needs some improvement. 🔧🔧 Keep experimenting! 💻💪<br>"

# Post a comment on the GitHub issue with the metrics
message = f"@{os.environ['GITHUB_ACTOR']} Hello, I trained the model! Here are the metrics:<br>{reply_message}"
issue_number = os.environ['ISSUE_NUMBER']
uri = os.environ['URI']
auth_header = f"Authorization: token {os.environ['GITHUB_TOKEN']}"
content_type_header = "Content-Type: application/json"
data = {"body": message, "type": "rich_text"}
data_str = json.dumps(data)
cmd = f"curl -X POST -sSL -d '{data_str}' -H '{auth_header}' -H '{content_type_header}' {uri}/repos/{os.environ['GITHUB_REPOSITORY']}/issues/{issue_number}/comments"
subprocess.check_call(cmd, shell=True)
