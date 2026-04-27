import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def train_3tier_model():
    # 1. Path to the 3-tier CSV
    data_path = os.path.join('..','datasets', '3tier_training_data.csv')
    
    if not os.path.exists(data_path):
        print("Error: 3tier_training_data.csv not found!")
        return

    # 2. Load and CLEAN the data
    df = pd.read_csv(data_path)
    
    # CRITICAL: Remove any rows that have empty text or empty labels
    df = df.dropna(subset=['text', 'label'])
    
    X = df['text'].astype('U')
    y = df['label'].astype(int) # Ensure labels are integers (0, 1, 2)

    print(f" Training 3-Tier Model on {len(X)} clean samples...")

    # 3. Vectorizer (The Context Fix)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_vec = vectorizer.fit_transform(X)

    # 4. Logistic Regression (Updated for modern sklearn)
    # Note: 'multi_class' is removed to avoid version errors
    model = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        class_weight='balanced'
    )
    
    print("Fitting model (this may take a moment)...")
    model.fit(X_vec, y)

    # 5. Save artifacts
    os.makedirs('../model_artifacts', exist_ok=True)
    joblib.dump(vectorizer, '../model_artifacts/vectorizer.pkl')
    joblib.dump(model, '../model_artifacts/toxicity_model.pkl')
    
    print(" 3-Tier Model Training Complete!")

if __name__ == "__main__":
    train_3tier_model()