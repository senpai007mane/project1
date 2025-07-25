import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns


print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)


df = pd.read_csv(f"{path}/diabetes.csv")

print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df['Outcome'].value_counts())

# Handle zero values (assuming they represent missing data)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("\nFeature coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

def predict_diabetes(new_data):
    """
    Predicts the probability of diabetes for new patient data.

    Args:
        new_data (dict or pandas.DataFrame): New patient data.

    Returns:
        tuple: A tuple containing the probability of diabetes and the prediction (0 for No Diabetes, 1 for Diabetes).
    """

    if isinstance(new_data, dict):
        new_df = pd.DataFrame([new_data])
    else:
        new_df = new_data.copy()

    required_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']
    for col in required_cols:
        if col not in new_df.columns:
            raise ValueError(f"Missing required column: {col}")


    scaled_data = scaler.transform(new_df[required_cols])

    probability = model.predict_proba(scaled_data)[:, 1]
    prediction = model.predict(scaled_data)[0]

    print("\nPrediction Results:")
    print(f"Probability of diabetes: {probability[0]:.4f}")
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")

    return probability[0], prediction


new_patient = {
    'Pregnancies': 2,
    'Glucose': 150,
    'BloodPressure': 70,
    'SkinThickness': 30,
    'Insulin': 100,
    'BMI': 35,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 40
}

predict_diabetes(new_patient)
