import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load your model and vectorizer
model = joblib.load('../model_artifacts/toxicity_model.pkl')
vectorizer = joblib.load('../model_artifacts/vectorizer.pkl')

# 2. ADDED: Load data and generate metrics
data_path = os.path.join('..','datasets', '3tier_training_data.csv')
if os.path.exists(data_path):
    # Load and clean as done in training
    df = pd.read_csv(data_path).dropna(subset=['text', 'label'])
    X = df['text'].astype('U')
    y = df['label'].astype(int)

    # Use the same 80/20 split and random_state as train_3.py
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the test set
    X_test_vec = vectorizer.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_vec)

    # Print the precision, recall, f1-score, and support
    print("\n" + "="*40)
    print("      MODEL PERFORMANCE METRICS")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['Tier 0', 'Tier 1', 'Tier 2']))
    print("="*40)
else:
    print("Warning: Dataset not found. Precision/Recall cannot be calculated.")

# 3. Existing Feature Visualization Code
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

word_scores = pd.DataFrame({'word': feature_names, 'score': coefficients})
top_toxic = word_scores.sort_values(by='score', ascending=True).head(15)
top_clean = word_scores.sort_values(by='score', ascending=False).head(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.barh(top_toxic['word'], top_toxic['score'], color='salmon')
ax1.set_title('Top "Toxic" Features (Tier 2 Triggers)')
ax1.invert_yaxis()
ax1.set_xlabel('Coefficient Weight')

ax2.barh(top_clean['word'], top_clean['score'], color='mediumseagreen')
ax2.set_title('Top "Safe" Features (Tier 0 Anchors)')
ax2.invert_yaxis()
ax2.set_xlabel('Coefficient Weight (Negative)')

plt.tight_layout()
plt.show()