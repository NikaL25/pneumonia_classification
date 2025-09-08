# 🩺 Pneumonia Detection from Chest X-ray

This is a Streamlit app that uses a trained deep learning model to detect pneumonia from chest X-ray images.

Live Demo: [https://nikal25-pneumonia-classification-app-hxbrmi.streamlit.app/](https://nikal25-pneumonia-classification-app-hxbrmi.streamlit.app/)

---

## Features

- Upload chest X-ray images (`jpg`, `jpeg`, `png`)
- Predict whether pneumonia is present
- Show confidence score
- Supports batch testing on local dataset folders

---

## 📁 Project Structure

app.py # Streamlit app
pneumonia_model.h5 # Trained Keras model (must be present)
requirements.txt # Python dependencies
venv/ # Virtual environment folder (optional, .gitignored)
chest_xray/ # Dataset folder for local testing
├── train/
├── test/
├── val/
├── NORMAL/
└── PNEUMONIA/
README.md # This file


---

## 🛠 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/NikaL25/pneumonia_classification.git
cd pneumonia_classification

2. Create a Python virtual environment (recommended)
python3 -m venv venv

Activate the virtual environment:

On macOS/Linux:
source venv/bin/activate

On Windows (cmd):
venv\Scripts\activate

3. Upgrade pip

pip install --upgrade pip


4. Install dependencies

pip install -r requirements.txt

5. Ensure the model file pneumonia_model.h5 is in the root directory

If missing, download or add it.

🚀 Running the App Locally

streamlit run app.py
