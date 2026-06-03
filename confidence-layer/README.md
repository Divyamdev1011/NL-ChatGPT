# Confidence Layer — Gemini Output Evaluation

A fully working web app that lets users ask Gemini anything, then analyzes every response in real time using a second Gemini call — surfacing signals, reasoning gaps, assumptions, and evaluation questions.

## Live Features

- **Ask anything** — real Gemini answers via Google Generative AI API
- **Inline highlights** — uncertain claims, things to verify, stated assumptions
- **Signals tab** — 2-4 specific signals per response with explanations  
- **Reasoning tab** — 4-step breakdown of what Gemini drew on and what it doesn't know
- **Evaluate tab** — guided questions to help you assess the output
- **Confidence bars** — per-claim confidence breakdown
- **Domain risk** — context-specific risk level per response
- **Calibration score** — tracks your evaluation engagement over the session
- **Toggle** — turn Confidence Layer on/off to see the difference

## Stack

- Python 3.10+
- Streamlit
- Google Generative AI Python SDK

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/confidence-layer
cd confidence-layer
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and paste your Gemini API key in the sidebar.

## Deploy to Streamlit Cloud (Free, Public URL)

1. Push this repo to GitHub (public or private)
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

You'll get a permanent URL like: `https://your-app-name.streamlit.app`

> No secrets needed in deployment — users enter their own Gemini API key in the sidebar.

## File Structure

```
confidence-layer/
├── app.py                  ← main Streamlit app
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← dark theme config
└── README.md
```
