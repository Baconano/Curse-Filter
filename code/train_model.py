import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def train_model():
    # 1. Path to the master dataset (one level up from the 'code' folder)
    data_path = os.path.join('..','datasets', 'master_training_dataset.csv')
    
    if not os.path.exists(data_path):
        print(f" Error: {data_path} not found. Run merge_datasets.py first!")
        return

    # 2. Load the pre-cleaned master data
    df = pd.read_csv(data_path)
    X = df['text'].astype('U') # Ensure all text is string format
    y = df['is_toxic']

    print(f"Training on {len(X)} combined samples...")
    print(f"Includes custom weights for insults and neutral objects.")

    
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 2)
    )
    X_vec = vectorizer.fit_transform(X)

    # 4. Logistic Regression
    # 'balanced' class_weight helps because there is more clean data than toxic data
    model = LogisticRegression(
        solver='liblinear', 
        class_weight='balanced', 
        max_iter=1000
    )
    model.fit(X_vec, y)

    # 5. Save Artifacts to the 'model_artifacts' folder
    os.makedirs('../model_artifacts', exist_ok=True)
    
    joblib.dump(vectorizer, '../model_artifacts/vectorizer.pkl')
    joblib.dump(model, '../model_artifacts/toxicity_model.pkl')
    
    print("Success! Your smarter model is trained and saved.")

if __name__ == "__main__":
    train_model()