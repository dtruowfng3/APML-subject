import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Kiểm tra GPU ---
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Using CPU instead.")

# --- Parameters ---
DATA_PATH = 'archive/IMDB Dataset.csv'
GLOVE_PATH = 'archive/glove.6B.100d.txt'
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 250
VOCAB_SIZE = 10000
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
EPOCHS = 5
BATCH_SIZE = 64

print("\n--- 1. Loading Data ---")
df = None
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Creating a dummy dataset...")
    data = {
        'review': [
            "This movie was absolutely fantastic! Loved every minute of it.",
            "A total waste of time, I hated it. Very boring.",
            "It was okay, not great, not terrible.",
            "Best movie ever! Highly recommend.",
            "Worst acting and plot. Don’t watch this."
        ],
        'sentiment': ["positive", "negative", "negative", "positive", "negative"]
    }
    df = pd.DataFrame(data)

# --- 2. Data Preprocessing ---
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['review'] = df['review'].apply(clean_text)

if len(df) < 100:
    tokenizer = Tokenizer(num_words=None, oov_token="<unk>")
else:
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")

tokenizer.fit_on_texts(df['review'])
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

sequences = tokenizer.texts_to_sequences(df['review'])

if len(df) < 100:
    current_max_len = max(len(s) for s in sequences) if sequences else 0
    actual_max_sequence_length = min(MAX_SEQUENCE_LENGTH, current_max_len)
    if actual_max_sequence_length == 0:
        actual_max_sequence_length = MAX_SEQUENCE_LENGTH
else:
    actual_max_sequence_length = MAX_SEQUENCE_LENGTH

padded_sequences = pad_sequences(sequences, maxlen=actual_max_sequence_length, padding='post', truncating='post')
labels = df['sentiment'].values

# --- 3. Load GloVe Embeddings ---
embeddings_index = {}
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
try:
    with open(GLOVE_PATH, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors in GloVe.")

    for word, i in word_index.items():
        if i < (len(tokenizer.word_index) + 1):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
except FileNotFoundError:
    print(f"Error: {GLOVE_PATH} not found. Using random embedding matrix.")
    embedding_matrix = np.random.rand(len(tokenizer.word_index) + 1, EMBEDDING_DIM)

# --- 4. Split Data ---
test_split_size = 0.4 if len(df) < 100 else 0.2
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=test_split_size, random_state=42, stratify=labels
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 5. Build Model ---
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
              input_length=actual_max_sequence_length, trainable=False),
    LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 6. Train Model (GPU nếu có) ---
current_epochs = 10 if len(df) < 100 else EPOCHS
current_batch_size = 2 if len(df) < 100 else BATCH_SIZE

with tf.device('/GPU:0'):   # 👈 ép chạy trên GPU nếu có
    history = model.fit(
        X_train, y_train,
        epochs=current_epochs,
        batch_size=current_batch_size,
        validation_split=0.1,
        verbose=1
    )

# --- 7. Evaluate ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# --- 8. Visualize Training ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend(); plt.grid(True)

plt.show()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.title("Confusion Matrix")
plt.show()

# --- 9. Analysis ---
tn, fp, fn, tp = cm.ravel()
print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
print("Recall:", tp / (tp + fn) if (tp+fn) else 0)
print("Precision:", tp / (tp + fp) if (tp+fp) else 0)
