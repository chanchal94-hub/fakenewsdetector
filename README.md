# 📰 Fake News Detector (Web App)

AI-powered web app jo news headline/article ko **REAL** ya **FAKE** classify karta hai, using a pre-trained Hugging Face model (`hamzab/roberta-fake-news-classification`).

## 🚀 Setup Instructions (Step-by-Step)

### 1. Python install karo
Python 3.9+ honi chahiye. Check karo:
```bash
python --version
```

### 2. Project folder me jao
```bash
cd fake-news-detector
```

### 3. Virtual environment banao (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Dependencies install karo
```bash
pip install -r requirements.txt
```
> Note: Pehli baar install hone me 5-10 min lag sakte hain (torch + transformers heavy hain).

### 5. App run karo
```bash
streamlit run app.py
```

Browser automatically khulega: `http://localhost:8501`

> Pehli baar run karne pe model download hoga (~500MB), internet chahiye. Uske baad cache ho jayega.

## 🧠 How it works

1. User headline + news text enter karta hai
2. Text ko model ke format me convert kiya jata hai
3. RoBERTa-based pre-trained model prediction deta hai: REAL ya FAKE + confidence %

## 📁 Files

- `app.py` — Main Streamlit app
- `requirements.txt` — Python dependencies

## 🌐 Free Deployment (Optional)

Tum is app ko free me deploy kar sakte ho:
1. GitHub pe code push karo
2. [share.streamlit.io](https://share.streamlit.io) pe jao
3. GitHub repo connect karo → Deploy

## ⚠️ Limitations

- Model English news ke liye best kaam karta hai
- 100% accuracy guarantee nahi — AI tool hai, final judgment hamesha human ka hona chahiye
- Bahut chhote text (1-2 words) pe accuracy kam ho sakti hai

## 🛠️ Future Improvements (Ideas)

- Multiple models compare karna (ensemble)
- URL se news scrape karke directly check karna
- History/database me past checks save karna
- Hindi/regional language support add karna
