import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# a) Read the data into a data frame.
data_path = '/workspaces/class/Data2/PassFail.dat'

df = pd.read_csv(data_path, sep=r'\s+', header=None)
df.columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']

# b) Determine the number of observations.
num_obs = df.shape[0]
print(f'Number of observations: {num_obs}')

# c) Split the data into training (60%) and testing (40%) datasets.
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)

# d) Train a logistic regression model using the training data.
features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X_train = train_df[features]
y_train = train_df['y']
X_test = test_df[features]
y_test = test_df['y']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# e) Score the test observations: get predicted probabilities for the positive class.
y_pred_prob = model.predict_proba(X_test)[:, 1]

# f) Compute the ROC curve and AUC.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f'ROC AUC score: {auc_score:.4f}')

# Plot, print, and save ROC curve.
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance.
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.grid(True)

output_dir = '/workspaces/class/Graphs2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_path = os.path.join(output_dir, 'roc_curve.png')
plt.savefig(plot_path)
plt.show()
print(f'ROC curve saved to {plot_path}')
