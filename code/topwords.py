import joblib
import pandas as pd
import numpy as np

# Load your newly trained artifacts
vectorizer = joblib.load('../model_artifacts/vectorizer.pkl')
model = joblib.load('../model_artifacts/toxicity_model.pkl')

# Get all the words from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get the weights (coefficients) from the model
# Higher positive numbers = Toxic | Lower negative numbers = Clean
coefficients = model.coef_[0]

# Create a DataFrame to view them easily
word_scores = pd.DataFrame({'word': feature_names, 'score': coefficients})

# Show top 20 Toxic words
print(" TOP TOXIC TRIGGER WORDS:")
print(word_scores.sort_values(by='score', ascending=True).head(20))

print("\n TOP CLEAN WORDS :")
print(word_scores.sort_values(by='score', ascending=False).head(20))