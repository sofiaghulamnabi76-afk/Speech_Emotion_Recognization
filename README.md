# 🎙️ Speech Emotion Recognition

Recognize human emotions (happy, sad, angry, calm, neutral, fearful, disgust, surprised) directly from speech audio using deep learning and speech signal processing.

Built as **Task 2: Emotion Recognition from Speech** — CODEALPHA INTERNSHIP

---

## 📌 Objective

Detect human emotions from raw speech audio by extracting acoustic features and classifying them using a neural network.

## 🧠 Approach

- **Feature Extraction:** MFCCs (Mel-Frequency Cepstral Coefficients) extracted from each audio clip using `librosa`
- **Model:** Fully connected Deep Neural Network (Dense layers with Dropout for regularization)
- **Framework:** TensorFlow / Keras

## 📂 Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,440 audio samples across 24 actors
- 8 emotion classes: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Source: [Kaggle - RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## 📊 Results

- **Test Accuracy:** 68.75%
- Training/validation accuracy curves saved as `accuracy_plot.png`

## 🗂️ Project Structure

```
speech-emotion-recognition/
├── emotion_recognition.ipynb   # Full training pipeline (feature extraction → training → evaluation → prediction)
├── app.py                      # Streamlit web app
├── emotion_model.h5            # Trained model
├── features.csv                # Extracted MFCC features + labels
├── accuracy_plot.png           # Training/validation accuracy graph
├── requirements.txt            # Python dependencies
└── data/                       # RAVDESS dataset (Actor_01 ... Actor_24) — not included, download separately
```

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the model (Jupyter Notebook)
Place the RAVDESS dataset inside a `data/` folder, then run `emotion_recognition.ipynb` cell by cell to extract features, train the model, and evaluate results.

### 2. Run the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` — upload a `.wav` file and get an instant emotion prediction with confidence scores.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `librosa` · `scikit-learn` · `Streamlit` · `Tkinter` · `NumPy` · `Pandas`

## 📈 Future Improvements

- Experiment with CNN/LSTM architectures on raw MFCC sequences (instead of averaged features)
- Expand training data with TESS and EMO-DB datasets
- Improve robustness to natural (non-scripted) speech
