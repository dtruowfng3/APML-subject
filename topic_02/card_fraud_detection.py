import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import time

print("--- Loading Data ---")
try: 
    df = pd.read_csv('archive/creditcard.csv')
    print("Data loaded successfully.")
    print("First 5 row of data")
    print(df.head())
    print(f"\nTontal number of transactions: {len(df)}")
    print("Class distribution (0: Not Fraud, 1: Fraud):")
    print(df['Class'].value_counts())
    print(f"Fraudulent transaction makes up {df['Class'].value_counts()[1] / len(df) * 100:.4f}% of the dataset.")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the file path is correct or dowload 'archive/creditcard.csv' from Kaggle.")
    exit()
    
    
X = df.drop('Class', axis=1)
y = df['Class']

print("\n--- Splitting data (Before Oversampling) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"Training set size: {X_train.shape} samples")
print(f"Test set size: {X_test.shape} samples")
print(f"Original training set class distribution:\n{y_train.value_counts()}")

print("\n--- Applying SMOTE to Training Data ---")
sm = SMOTE(random_state=42)
start_time_smote = time.time()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)    
smote_time = time.time() - start_time_smote
print(f"SMOTE execution time: {smote_time:.4f} seconds")

print(f"Resampled training set size: {X_train_res.shape[0]} samples")
print(f"Resampled training set class distribution:\n{y_train_res.value_counts()}")


print("\n--- Training Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
start_time_train = time.time()
model.fit(X_train_res, y_train_res)
train_time = time.time() - start_time_train
print(f"Random Forest model trained successfully in {train_time:.4f} seconds.")


print("\n--- Evaluating Model ---")
start_time_predict = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - start_time_predict
print(f"Prediction time: {predict_time:.4f} seconds")


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")    
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


print("Classification Report:")
print(classification_report(y_test, y_pred))


print(f"Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Fraud (0)', 'Predicted Fraud (1)'],
            yticklabels=['Actual Not Fraud (0)', 'Actual Fraud (1)'])
plt.title('Confusion Matrix for Credit Card Fraud Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Phân tích confusion matrix
print("\n--- Analysis of Results (from Confusion Matrix) ---")
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly identified Not Fraud): {tn}")
print(f"False Positives (Not Fraud -> Fraud): {fp} (These are 'False Alarms')")
print(f"False Negatives (Fraud -> Not Fraud): {fn} (These are 'Missed Frauds' - very critical!)")
print(f"True Positives (Correctly identified Fraud): {tp}")