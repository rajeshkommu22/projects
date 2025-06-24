import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


# Unzip manually
with zipfile.ZipFile('/content/statlog+german+credit+data.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/german_data')

# Load the correct file
# "german.data" has categorical values and needs column names
column_names = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "ResidenceSince", "Property", 
    "Age", "OtherInstallmentPlans", "Housing", "ExistingCredits", "Job", "LiablePeople", 
    "Telephone", "ForeignWorker", "Risk"
]

df = pd.read_csv('/content/german_data/german.data', sep=' ', header=None, names=column_names) #i'm using zip file u can upload direct credit scoring model dataset

print(df.head())
print("Missing Values Per Column:")

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
df.shape
df.describe().T


le = LabelEncoder()
df['Risk'] = le.fit_transform(df['Risk'])


# Encode all categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print(df['Risk'].value_counts())
df.head()
X = df.drop('Risk', axis=1)
y = df['Risk']





# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_cols = ['Duration', 'CreditAmount', 'InstallmentRate', 'ResidenceSince',
                  'Age', 'ExistingCredits', 'LiablePeople']


scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#Step 5: Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight = 'balanced')
lr_model.fit(X_train, y_train)

#Prediction & Evaluation
y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:, 1]

# Scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Evaluation Metrics:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})", color="green")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()





