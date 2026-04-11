# Speech Emotion Recognition

A deep learning system that recognizes emotions and gender from audio files using CNN models trained on multiple datasets.

## Features
- **Emotion Recognition**: 6 emotions (fear, angry, neutral, happy, sad, surprise)
- **Gender Detection**: Male/Female classification
- **High Accuracy**: 91.1% emotion, 98.6% gender
- **Web Interface**: Interactive Streamlit app
- **Real-time Processing**: Upload audio and get instant results

## Tech Stack
- **Python** - Core language
- **TensorFlow** - Deep learning framework
- **Streamlit** - Web interface
- **Librosa** - Audio processing
- **NumPy** - Numerical operations

## Quick Start

```bash
git clone https://github.com/Thunnpoe/speech-emotion-recognition.git
cd speech-emotion-recognition
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
speech-emotion-recognition/
|-- app.py                    # Main application
|-- model3_self_trained.h5      # Emotion model
|-- model_mw_self_trained.h5     # Gender model
|-- df_audio.csv              # Dataset metadata
|-- datasets/                 # Audio files
|-- requirements.txt           # Dependencies
|-- train_from_scratch.py       # Training script
```

## Usage
1. Open web app
2. Upload audio file (WAV, MP3, M4A)
3. Get emotion and gender predictions
4. View confidence scores and visualizations

## Models
- **Datasets**: CREMA-D, RAVDESS, SAVEE, TESS (12,162 samples)
- **Architecture**: CNN with MFCC features
- **Performance**: 91.1% emotion accuracy, 98.6% gender accuracy


---

**GitHub Repository**: https://github.com/Thunnpoe/speech-emotion-recognition

Star this repo if it helps you!
