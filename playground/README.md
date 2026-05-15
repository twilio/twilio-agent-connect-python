# TAC Playground

Interactive web playground for testing Twilio Agent Connect.

## Deploy Options

### Option 1: Streamlit Community Cloud (Recommended - Free)

1. Fork this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Point to `playground/app.py`
5. Deploy!

Users visit your Streamlit app URL, paste credentials, and test instantly.

### Option 2: Hugging Face Spaces

1. Create a new Space: [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose "Streamlit" as SDK
3. Upload `playground/app.py` and `requirements.txt`
4. Set Python version to 3.11+

### Option 3: Railway

```bash
cd playground
railway init
railway up
```

## Local Testing

```bash
cd playground
pip install -r requirements.txt
streamlit run app.py
```

Then use ngrok for webhooks:
```bash
ngrok http 8501
```

## How It Works

1. Users enter their credentials in the sidebar
2. App creates a temporary `.env` file
3. Starts the TAC FastAPI server with those credentials
4. Shows webhook URLs to configure in Twilio Console
5. Displays live server logs

## Environment Variables

The deployed playground needs access to the TAC package. Options:

- **Streamlit Cloud:** Add GitHub repo, it will auto-detect `pyproject.toml`
- **Hugging Face:** May need to create a `packages.txt` with system deps
- **Railway:** Auto-detects Python and installs from `pyproject.toml`
