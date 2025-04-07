import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Load dataset
file_path = "data/DiseaseAndSymptoms.csv"
df = pd.read_csv(file_path)

# 🔹 Combine all symptom columns into a single text column
df["Symptoms"] = df.iloc[:, 1:].fillna("").apply(lambda x: " ".join(x).strip(), axis=1)

# 🔹 Convert text symptoms to numerical format using TF-IDF (Optimized)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # Reduce complexity by limiting to bigrams
    max_df=0.9,            # Ignore very frequent terms
    min_df=2,              # Ignore very rare terms
    max_features=5000,     # Reduce feature set size
    sublinear_tf=True,     # Prevent over-weighting frequent words
    stop_words='english'   # Remove common English stopwords
)
X = vectorizer.fit_transform(df["Symptoms"])
y = df["Disease"]

# 🔹 Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

# 🔹 Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 🔹 Cross-validation to check real performance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
print(f"✅ Cross-validation accuracy: {cv_scores.mean():.4f}")

# 🔹 Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 🔹 Print short classification report
print("\n📌 Model: Naïve Bayes (TF-IDF)")
print(f"✅ Test Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
print(f"✅ Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1-score: {report['weighted avg']['f1-score']:.4f}")

# 🔹 Save model & vectorizer using joblib
try:
    joblib.dump(model, "models/naive_bayes_text_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("✅ Model & vectorizer saved successfully!")
except Exception as e:
    print(f"❌ Error saving model or vectorizer: {e}")
