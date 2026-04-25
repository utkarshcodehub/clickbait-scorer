import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
import re

def extract_features(headline):
    """Hand-crafted clickbait signals"""
    h = headline.lower()
    return {
        'has_number': bool(re.search(r'\d+', headline)),
        'all_caps_words': sum(1 for w in headline.split() if w.isupper() and len(w) > 2),
        'has_question': '?' in headline,
        'has_exclamation': '!' in headline,
        'word_count': len(headline.split()),
        'has_you': 'you' in h or 'your' in h,
        'has_shocking': any(w in h for w in ['shock', 'stun', 'amaz', 'unbeliev', 'reveal', 'secret', 'truth', 'real reason', 'dark', 'exposes']),
        'has_superlative': any(w in h for w in ['best', 'worst', 'most', 'least', 'never', 'always', 'every', 'only']),
        'has_will': 'will' in h,
        'has_how_why': headline.lower().startswith(('how', 'why', 'what')),
        'headline_len': len(headline),
        'has_ellipsis': '...' in headline,
    }

# Load data
dfs = []
try:
    dfs.append(pd.read_csv('headlines_raw.csv')[['headline', 'label']])
    print(f"Scraped data: {len(dfs[-1])} rows")
except:
    print("No scraped data found, using seed only")

dfs.append(pd.read_csv('seed_data.csv'))
df = pd.concat(dfs).drop_duplicates('headline').dropna()
print(f"Total training data: {len(df)} rows | Label distribution:\n{df['label'].value_counts()}")

# Feature engineering
feature_df = pd.DataFrame([extract_features(h) for h in df['headline']])

# TF-IDF on text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
tfidf = vectorizer.fit_transform(df['headline'])

# Combine hand features + TF-IDF
import scipy.sparse as sp
X = sp.hstack([tfidf, sp.csr_matrix(feature_df.values.astype(float))])
y = df['label'].values

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Save
joblib.dump(model, 'clickbait_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(list(feature_df.columns), 'feature_cols.pkl')
print("\nModel saved.")

# Quick test
def score(headline):
    feats = pd.DataFrame([extract_features(headline)])
    tfidf_vec = vectorizer.transform([headline])
    x = sp.hstack([tfidf_vec, sp.csr_matrix(feats.values.astype(float))])
    prob = model.predict_proba(x)[0][1]
    return prob

test_headlines = [
    "You Won't BELIEVE What This IIT Student Did",
    "RBI Keeps Repo Rate Unchanged at 6.5%",
    "10 Shocking Reasons Why India Will Never Change",
    "Supreme Court Stays Order on Electoral Bonds"
]
print("\n--- Quick test ---")
for h in test_headlines:
    print(f"{score(h):.2f}  →  {h}")