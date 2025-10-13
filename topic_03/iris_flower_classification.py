import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    df_iris = pd.read_csv('archive/IRIS.csv')
except FileNotFoundError:
    print("File 'iris_dataset.csv' not found. Using scikit-learn’s built-in Iris dataset.")
    iris = load_iris()
    df_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])
    df_iris['species'] = df_iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df_iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("--- 1. Load and Explore Data ---")
print(df_iris.head())
print(f"Data shape: {df_iris.shape}\n")

X = df_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 2. Training and Test Set Sizes ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

print("--- 3. KNN Model Training Complete ---\n")

y_pred = knn.predict(X_test)

print("--- 4. Prediction Results on Test Set ---")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head())
print("\n")

cm = confusion_matrix(y_test, y_pred)
print("--- 5. Confusion Matrix ---")
print(cm)
print("\n")

accuracy = accuracy_score(y_test, y_pred)
print("--- 6. Model Accuracy ---")
print(f"Accuracy: {accuracy:.2f}\n")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Iris Classification with KNN')
plt.show()

print("--- 7. Results Analysis ---")
if accuracy > 0.95:
    print("KNN model achieved very high accuracy, showing good classification capability for Iris species.")
elif accuracy > 0.8:
    print("KNN model achieved good accuracy, which may be acceptable for this problem.")
else:
    print("KNN model might need further optimization (e.g., changing n_neighbors, data normalization).")

print(f"\nNumber of correctly classified samples: {np.trace(cm)}")
print(f"Number of misclassified samples: {np.sum(cm) - np.trace(cm)}")

print("\n--- 8. Prediction for a New Iris Sample ---")
new_flower_features = pd.DataFrame([[5.0, 3.5, 1.3, 0.2]],
                                   columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
predicted_species = knn.predict(new_flower_features)
print(f"New flower features: Sepal Length={new_flower_features.iloc[0,0]}, "
      f"Sepal Width={new_flower_features.iloc[0,1]}, "
      f"Petal Length={new_flower_features.iloc[0,2]}, "
      f"Petal Width={new_flower_features.iloc[0,3]}")
print(f"Predicted species: {predicted_species[0]}")
