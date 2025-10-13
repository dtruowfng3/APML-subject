import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    
print("---Load and Explore Data---")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
df.columns = ['label', 'message']

print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nGeneral info about the dataset:")
df.info()
print("\nLabel distribution: (spam/ham):")
print(df['label'].value_counts())
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nCreated numerical label column 'label_num'(0: ham, 1: spam).")

print("\n---Text Preprocessing---")
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text =re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    return " ".join(filtered_words)

df['clean_message'] = df['message'].apply(preprocess_text)
print("Text preprocessing completed.")
print("\nOriginal: ", df['message'][0])
print("Processed: ", df['clean_message'][0])

print("\n---Split Data---")
X = df['clean_message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

print("\n---Text Vectorization---")
vectorizer = CountVectorizer()
# vectorizer = CountVectorizer(min_df=0.003, max_df=2000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
print(f"Training matrix shape: {X_train_vect.shape}")

print("\n---Model Training---")
model = MultinomialNB(alpha=1.0) #applies Laplace smoothing
model.fit(X_train_vect, y_train)
print("Model trained successfully.")

print("\n---Model Evaluation---")
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham (0)', 'Spam (1)'], yticklabels=['Ham (0)', 'Spam (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (Correctly Ham): {tn}")
print(f"False Positives (Ham -> Spam): {fp}")
print(f"False Negatives (Spam -> Ham): {fn}")       
print(f"True Positives (Correctly Spam): {tp}")