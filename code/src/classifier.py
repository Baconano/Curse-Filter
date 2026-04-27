import joblib
import os

class ToxicityClassifier:
    def __init__(self):
        # Locate the artifacts folder relative to this file
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '../../model_artifacts/toxicity_model.pkl')
        vec_path = os.path.join(base_path, '../../model_artifacts/vectorizer.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifacts not found. Run train_3tier.py first!")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)

    def predict_tier(self, text):
        cleaned_text = text.lower().strip()
        words = cleaned_text.split()

        # Hard-coded Neutral 'You' Shield
        if len(words) <= 2 and "you" in words:
            return "Clean", 1.0

        if not cleaned_text:
            return "Clean", 0.0

        vec_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vec_text)[0]
        probs = self.model.predict_proba(vec_text)[0]
        confidence = max(probs)

        # Map numbers back to names
        status_map = {0: "Clean", 1: "Tier 1", 2: "Tier 2"}
        status = status_map.get(prediction, "Clean")

        # Confidence logic override
        if status == "Tier 2" and confidence < 0.75:
            status = "Tier 1"
            
        return status, confidence