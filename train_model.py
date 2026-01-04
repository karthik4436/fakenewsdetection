import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
true_path = r"C:\Users\karth\OneDrive\Documents\fake_news_detection\dataset\True.csv"
fake_path = r"C:\Users\karth\OneDrive\Documents\fake_news_detection\dataset\Fake.csv"

df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake])
df = df.sample(frac=1).reset_index(drop=True)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save files
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully")
