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
```

## Setup and Execution

### Prerequisites
* **Python 3.10+**: The core environment for execution.
* **FFmpeg**: A mandatory dependency for the OpenAI Whisper engine to process acoustic signals.
* **Discord Bot Token**: Required for the system to interface with the Discord API. This should be stored in an `.env` file or configured within `main.py`.

### Installation
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd project

### 2. Installation & Dependency Setup

To prepare the environment for the Toxicity Moderation System, follow these steps:

1. **System-Level Dependencies:** Ensure **FFmpeg** is installed on your machine. This is required by the Whisper STT engine for audio decoding.
   * **Windows (Chocolatey):** `choco install ffmpeg`
   * **macOS (Homebrew):** `brew install ffmpeg`
   * **Linux (Ubuntu):** `sudo apt install ffmpeg`

2. **Python Environment:** It is recommended to use a virtual environment (venv) to avoid dependency conflicts.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate\
### 3. Core Python Libraries

The system relies on the following key libraries to manage the real-time moderation pipeline:

* **`scikit-learn`**: Powers the **Multinomial Logistic Regression** engine and the **TF-IDF Vectorization** used to categorize text into the three toxicity tiers.
* **`openai-whisper`**: A robust Speech-to-Text (STT) model used to transcribe live audio streams from Discord voice channels into structured text for analysis.
* **`PyAudio`**: Facilitates the low-level system audio monitoring, allowing the bot to capture real-time acoustic signals from the user's local environment.
* **`discord.py`**: The primary API wrapper used to interface with Discord, enabling the bot to execute moderation actions like temporary mutes and warnings.
* **`joblib`**: Utilized for model persistence, allowing the system to save and load the trained `.pkl` classifier and vectorizer files efficiently.

To install the entire stack, run:
```bash
pip install -r code/requirements.txt