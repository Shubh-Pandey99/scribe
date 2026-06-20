# Scribe

> AI-powered live transcription, screen capture analysis, and real-time Q&A — right in your browser sidepanel.

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

👉 **Live:** [scribe-extension.vercel.app](https://scribe-extension.vercel.app)

</div>

## What it does

- 🎙️ **Live Transcription** — Captures audio from any browser tab or microphone and transcribes it in real-time using Groq Whisper, OpenAI Whisper, or Google Gemini.
- 📷 **Screen Capture & Paste** — Takes a screenshot of the active tab (or paste an image directly) and sends it to Gemini for visual analysis.
- 🤖 **AI Q&A** — Ask anything about the transcript or captures via the command bar.
- ✨ **Summarize** — One-click summaries in bullet points.
- 🕐 **Session History** — Every session saved locally and synced to the cloud.
- 👁️ **Stealth Mode** — Blurs the panel in high-pressure environments. Hover to reveal.

## Architecture

```
Chrome Extension (MV3)          Vercel Backend (Flask)
─────────────────────           ──────────────────────
sidepanel.html/js/css    ──►   /api/transcribe   (Groq → Whisper → Gemini STT)
background-enhanced.js   ──►   /api/answer       (Gemini text + vision)
content-script.js        ──►   /api/sessions     (PostgreSQL persistence)
manifest.json
```

## Setup

### 1. Extension
1. Clone this repo.
2. Go to `chrome://extensions/` → Enable **Developer Mode** → **Load Unpacked** → select this folder.
3. Open any tab → click the Extensions puzzle icon → **Scribe** → Open sidepanel.

### 2. Backend (Vercel)
Deploy the `api/` directory to Vercel:
```bash
vercel deploy
```
Set these environment variables in the Vercel dashboard:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Lightning-fast Whisper STT (primary) |
| `OPENAI_API_KEY` | Whisper STT (fallback) |
| `GOOGLE_API_KEY` | Gemini multimodal AI + STT fallback |
| `SCRIBE_API_KEY` | API authentication key |
| `POSTGRES_URL` | Session cloud sync (optional) |

### 3. Connect
In the Scribe sidepanel → Settings (⚙️):
- **API URL**: your Vercel deployment URL
- **API Key**: your `SCRIBE_API_KEY` value

## Usage

1. Open any tab with audio (YouTube, Zoom, Google Meet, etc.)
2. Click **Tab** to record tab audio, or **Mic** to record microphone.
3. The live transcript appears as audio is processed.
4. Hit **Summarize** for bullet-point notes, or type a question in the command bar.
5. Use **Capture** to screenshot the current tab and get visual AI analysis.

## Tech Stack

- **Extension**: Vanilla JS, Chrome MV3, Web Audio API, `getDisplayMedia` + microphone capture
- **Backend**: Python Flask on Vercel
- **STT**: Groq Whisper (`whisper-large-v3-turbo`) → OpenAI Whisper → Gemini multimodal (fallback chain)
- **AI**: Google Gemini 2.5 Flash (default), 2.5 Pro, 2.0 Flash
- **Storage**: `chrome.storage.local` + PostgreSQL (`pg8000`)
- **UI**: Deep space glassmorphism, Inter + JetBrains Mono, CSS animations