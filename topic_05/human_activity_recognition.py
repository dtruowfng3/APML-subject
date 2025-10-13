import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


base_dir = "C:/Users/TRUONGVO/OneDrive/Desktop/APML/topic_05/archive/UCI HAR Dataset"

# def load_data(X_path, y_path):
#     X = pd.read_csv(X_path, delim_whitespace=True, header=None)
#     y = pd.read_csv(y_path, delim_whitespace=True, header=None)[0]
#     return X, y
def load_data(X_path, y_path):
    X = pd.read_csv(X_path, sep=r"\s+", header=None)
    y = pd.read_csv(y_path, sep=r"\s+", header=None)[0]
    return X, y


X_train_path = f"{base_dir}/train/X_train.txt"
y_train_path = f"{base_dir}/train/y_train.txt"
X_test_path  = f"{base_dir}/test/X_test.txt"
y_test_path  = f"{base_dir}/test/y_test.txt"
activity_labels_path = f"{base_dir}/activity_labels.txt"
features_path = f"{base_dir}/features.txt"

X_train, y_train = load_data(X_train_path, y_train_path)
X_test,  y_test  = load_data(X_test_path, y_test_path)

activity_labels = pd.read_csv(activity_labels_path, sep=r'\s+', header=None, names=['id', 'label'])
features = pd.read_csv(features_path, sep=r'\s+', header=None, names=['id', 'name'])

X_train.columns = X_test.columns = features['name'].values

y_train_names = y_train.map(dict(zip(activity_labels['id'], activity_labels['label'])))
y_test_names  = y_test.map(dict(zip(activity_labels['id'], activity_labels['label'])))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# models = {
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
#     "SVM": SVC(random_state=42),
#     "Naive Bayes": GaussianNB(),
#     "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
#     "KNN": KNeighborsClassifier(n_neighbors=5)
# }

# models = {
#     "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
#     "SVM": SVC(random_state=42, class_weight='balanced'),
#     "Naive Bayes": GaussianNB(),
#     "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
#     "KNN": KNeighborsClassifier(n_neighbors=5)
# }


results = {}

for name, model in models.items():
    start = time.time()
    if name in ["SVM", "MLP", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    end = time.time()

    # Tính các chỉ số (weighted để xử lý imbalance phần nào)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "time": end - start
    }

    # In báo cáo chi tiết cho mô hình
    print("\n" + "="*40)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred, target_names=activity_labels['label'].values, zero_division=0))

    # Vẽ confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activity_labels['label'].values,
                yticklabels=activity_labels['label'].values)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Tổng hợp kết quả
df = pd.DataFrame(results).T
print("\n=== Summary (sorted by accuracy) ===")
print(df.sort_values("accuracy", ascending=False))

# Vẽ biểu đồ so sánh
metrics_plot = df[['accuracy','precision','recall','f1']].astype(float)
metrics_plot.plot(kind='bar', figsize=(12,6), title="Model Performance")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Thời gian huấn luyện
df['time'].plot(kind='bar', figsize=(10,5), title="Training Time (seconds)")
plt.ylabel("Seconds")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
