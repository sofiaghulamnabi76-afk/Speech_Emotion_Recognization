import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file and the model will predict its emotion.")

# Load model
model = load_model("emotion_model.h5")

# Recreate scaler and label encoder from training data
df = pd.read_csv("features.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
le.fit(y)

scaler = StandardScaler()
scaler.fit(X)

def extract_mfcc(file_path, n_mfcc=40):
    y_audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

uploaded_file = st.file_uploader("Choose an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict Emotion"):
        mfcc = extract_mfcc("temp.wav").reshape(1, -1)
        mfcc_scaled = scaler.transform(mfcc)
        prediction = model.predict(mfcc_scaled)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        st.success(f"Predicted Emotion: **{predicted_label.upper()}**")

        st.subheader("Confidence Scores")
        for label, prob in zip(le.classes_, prediction[0]):
            st.write(f"{label}: {prob*100:.2f}%")