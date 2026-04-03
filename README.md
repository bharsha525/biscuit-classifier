# 🍪 Biscuit Wrapper Classifier

An AI-powered web application that classifies biscuit wrapper images into their respective brands using a Convolutional Neural Network (CNN).

## 🚀 Live Demo
👉 [Click here to try the app](https://biscuit-classifier.onrender.com/)

---

## 📌 About the Project

This project builds a deep learning model capable of automatically identifying biscuit brands from their packaging images. It serves as a tool for:

- 🛒 **Market researchers** to analyze design trends and consumer preferences
- 🎨 **Packaging designers** to draw inspiration from diverse wrapper styles
- 🤖 **AI enthusiasts** to explore real-world image classification

---

## 🧠 How It Works

1. User uploads a biscuit wrapper image
2. The image is preprocessed and resized to 224x224
3. A CNN model predicts the brand from 100 biscuit categories
4. The app displays the predicted brand with confidence score

---

## 🗂️ Project Structure
biscuit-classifier/
│
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── style.css           # Styling
└── biscuit.ipynb           # Model training notebook
---

## 🏷️ Supported Biscuit Brands (100 Classes)

Includes brands like:
- Britannia, Parle, Sunfeast, McVities
- Cadbury, UNIBIC, Cremica, Priyagold
- Sri Sri Tattva, Patanjali, Bonn, and many more!

---

## 🛠️ Tech Stack

| Layer | Technology |

| Frontend | HTML, CSS |
| Backend | Python, Flask |
| ML Model | CNN (TensorFlow/Keras) |
| Model Format | TFLite (quantized) |
| Hosting | Render |

---

## ⚙️ Run Locally

### 1. Clone the repository
git clone https://github.com/bharsha525/biscuit-classifier.git
cd biscuit-classifier
2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
3. Install dependencies
pip install -r requirements.txt
4. Run the app
python app.py
5. Open in browser
http://localhost:5000


📦 Model Details
Architecture: Convolutional Neural Network (CNN)
Input Size: 224 x 224 pixels
Output Classes: 100 biscuit brands
Original Model Size: 2.15 GB (.h5)
Optimized Model Size: 21.2 MB (.tflite with quantization)
Model Hosted: Google Drive (auto-downloaded on first run)


📸 Screenshots










































































































































































