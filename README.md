# Scribe

> AI-powered live transcription, screen capture analysis, and real-time Q&A — right in your browser sidepanel.

## What it does

- **GitHub**: [github.com/Shubh-Pandey99/scribe](https://github.com/Shubh-Pandey99/scribe)
- **Vercel**: [scribe-extension.vercel.app](https://scribe-extension.vercel.app)

- 🎙️ **Live Transcription** — Captures audio from any browser tab or **microphone** and transcribes it in real-time using Groq Whisper, OpenAI Whisper, or Google Gemini.
- 📷 **Screen Capture & Native Paste** — Takes a screenshot of the active tab (or paste an image directly) and sends it to Gemini for visual analysis.
- 🤖 **AI Q&A** — Ask anything about the transcript or capture using Gemini 2.0 Flash.
- ✨ **Summarize** — One-click meeting/video summaries in bullet points.
- 🕐 **Session History** — Every recording is saved locally and synced remotely.
- 👁️ **Stealth Mode** — Toggle stealth mode to blur UI elements in high-pressure environments.

## Architecture

```
Chrome Extension (MV3)          Vercel Backend
─────────────────────           ──────────────
sidepanel.html/js/css    ──►   /api/transcribe  (Groq → Whisper → Gemini STT)
background-enhanced.js   ──►   /api/answer      (Gemini text/vision)
manifest.json            ──►   /api/sessions    (PostgreSQL persistence)
```

## Setup

### Extension
1. Clone this repo.
2. Go to `chrome://extensions/` → Enable Developer Mode → Load Unpacked → select this folder.
3. Open any tab, click the Extensions puzzle icon → **Scribe** → Open sidepanel.

### Backend (Vercel)
1. Deploy the `api` directory to Vercel: `vercel deploy`
2. Set environment variables in Vercel dashboard:
   - `GROQ_API_KEY` — for lightning-fast Whisper STT (Primary)
   - `OPENAI_API_KEY` — for Whisper STT (Fallback)
   - `GOOGLE_API_KEY` — for Gemini Multimodal STT + AI responses
   - `POSTGRES_URL` — for session cloud sync using PostgreSQL (optional)

### Using Live Transcription
1. Open a YouTube video, a meeting, or simply use your microphone.
2. Click **Record Tab** or **Record Mic** in the sidepanel.
3. If recording a tab, Chrome will show a screen/tab share picker — select your tab and **check "Share tab audio"**.
4. Transcription appears live as audio is processed in fast audio chunks.

## Tech Stack

- **Extension**: Vanilla JS, MV3, `getDisplayMedia` & microphone capture, Context-aware prompting
- **Backend**: Python Flask on Vercel
- **STT**: Groq Whisper (`whisper-large-v3-turbo`) with OpenAI and Gemini multimodal fallbacks
- **AI**: Google Gemini (2.0 Flash, 1.5 Flash fallbacks)
- **Storage**: `chrome.storage.local` + PostgreSQL (`pg8000`)