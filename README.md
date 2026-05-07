# Toxicity Classification & Automated Moderation System

**Author:** Steven Khang Nguyen  
**Student ID:** skn1478  
**Institution:** University of Texas at Arlington  
**Course:** CSE 4392 Natural Language Processing  

---

## 1. Project Overview
This project addresses the challenge of real-time moderation in digital social environments, specifically prototyped for **Discord**. Unlike traditional binary "toxic/non-toxic" filters, this system utilizes a **three-tier classification framework** to distinguish between different levels of linguistic severity.

### Toxicity Tiers:
* **Tier 0 (General):** Conversational speech requiring no intervention.
* **Tier 1 (Warning/Mute):** Casual profanity or non-targeted emphasis.
* **Tier 2 (Administrative Action):** Severe hate speech, threats, or harassment.

The system integrates a **Whisper-based Speech-to-Text (STT)** pipeline, allowing the moderation of voice channels by transcribing real-time audio before classification.

---

## 2. Technical Methodology
* **Model Architecture:** Multinomial Logistic Regression utilizing a Softmax function for probabilistic tier assignment.
* **Feature Engineering:** TF-IDF Vectorization with an n-gram range of (1, 2).
* **Dataset:** A hybridized corpus of ~50,000 samples merged from the Jigsaw Toxic Comment and Davidson Hate Speech datasets.
* **Bias Mitigation:** Strategic injection of 5,000 neutral anchor samples (e.g., "you", "are", "how") to reduce false-positive rates on common pronouns.

---

## 3. Repository Structure
```text
project/
├── code/
│   ├── src/
│   │   ├── audio_processor.py  # Whisper STT processing logic
│   │   ├── classifier.py       # Toxicity Classification class
│   │   └── local_monitor.py    # PyAudio system audio listener
│   ├── main.py                 # Discord Bot orchestrator (entry point)
│   ├── 3_tier.py               # Dataset hybridization logic
│   ├── train_3.py              # Model training and .pkl persistence
│   └── requirements.txt        # Python dependencies
├── data/
│   ├── model.pkl               # Serialized Logistic Regression model
│   └── vectorizer.pkl          # Serialized TF-IDF Vectorizer
└── paper/
    ├── Nguyen_skn1478_Paper.pdf # Final research paper
    └── source_tex/             # LaTeX source files