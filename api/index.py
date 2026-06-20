import os, io, json, time, base64, logging, tempfile
from functools import wraps
from abc import ABC, abstractmethod
from io import BytesIO

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

import requests
from PIL import Image, UnidentifiedImageError

from openai import OpenAI
import google.generativeai as genai

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scribe-api")

def retry_with_backoff(max_retries=3, base_delay=1):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **k):
            for i in range(max_retries):
                try: return fn(*a, **k)
                except Exception as e:
                    if i == max_retries-1: log.error("Final attempt failed for %s: %s", fn.__name__, e); raise
                    time.sleep(base_delay * (2 ** i))
        return wrap
    return deco

# ---------- Providers ----------
class BaseProvider(ABC):
    @abstractmethod
    def get_response(self, transcript=None, image_url=None, image_base64=None, image_array=None): ...

    @abstractmethod
    def stream_response(self, transcript): ...

class OpenAIProvider(BaseProvider):
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise ValueError("OPENAI_API_KEY not configured")
        self.client = OpenAI(api_key=key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    @retry_with_backoff()
    def get_response(self, transcript=None, image_url=None, image_base64=None):
        messages = []
        if transcript and (image_url or image_base64):
            content = [{"type": "text", "text": transcript}]
            if image_base64:
                content.append({"type":"image_url","image_url":{"url": image_base64}})
            elif image_url:
                content.append({"type":"image_url","image_url":{"url": image_url}})
            messages.append({"role":"user","content":content})
        elif transcript:
            messages.append({"role":"user","content":transcript})
        else:
            if not (image_url or image_base64): return {"error":"No input provided"}
            content = [{"type":"text","text":"Analyze this image and summarize key insights."}]
            if image_base64:
                content.append({"type":"image_url","image_url":{"url": image_base64}})
            else:
                content.append({"type":"image_url","image_url":{"url": image_url}})
            messages.append({"role":"user","content":content})

        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return {"answer": resp.choices[0].message.content}

    def stream_response(self, transcript):
        yield f"data: {json.dumps({'error':'OpenAI streaming not used here'})}\n\n"

class GoogleProvider(BaseProvider):
    def __init__(self):
        key = os.getenv("GOOGLE_API_KEY")
        if not key: raise ValueError("GOOGLE_API_KEY not configured")
        genai.configure(api_key=key, transport="rest")
        self.model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        log.info("Initializing GoogleProvider with model: %s", self.model_name)
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=(
                "You are Scribe, an elite universal interview assistant and expert co-pilot embedded in a browser sidepanel. "
                "Your primary purpose is to help the user answer complex questions and solve problems across ANY domain (e.g., Software Engineering, Teaching, Government Exams, Finance, Law, etc.).\n\n"
                "CRITICAL Directives:\n"
                "1. Read the provided transcript context and any attached screenshots deeply to understand exactly what is being asked.\n"
                "2. Provide highly accurate, comprehensive, and well-thought-out answers. Do not make the answer so short that it loses critical nuance or context.\n"
                "3. If it is a coding question, provide the optimal working code with a Time/Space complexity breakdown.\n"
                "4. If it is a behavioral or scenario-based question (e.g., a teaching scenario or policy question), write out the ideal, comprehensive talking points the user should say in response.\n"
                "5. While you should be comprehensive, format your answer powerfully so the user can skim it while speaking. Use bolding for key terms, clear paragraphs, and bullet points where appropriate."
            )
        )

    def _pil_from_base64(self, data_uri:str):
        header, encoded = data_uri.split(",",1)
        b = base64.b64decode(encoded)
        try:
            return Image.open(BytesIO(b))
        except UnidentifiedImageError:
            raise ValueError("Invalid image data")

    def _pil_from_url(self, url:str):
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code != 200: raise ValueError(f"Image download failed HTTP {r.status_code}")
        try: return Image.open(BytesIO(r.content))
        except UnidentifiedImageError: raise ValueError("Failed to decode image")

    @retry_with_backoff()
    def get_response(self, transcript=None, image_url=None, image_base64=None, image_array=None):
        parts = []
        if transcript: parts.append(transcript)
        
        if image_array:
            for b64 in image_array:
                parts.append(self._pil_from_base64(b64))
        elif image_base64: parts.append(self._pil_from_base64(image_base64))
        elif image_url:  parts.append(self._pil_from_url(image_url))
        if not parts: return {"error":"No input provided"}
        
        try:
            log.info("Generating content for model %s (parts: %d)", self.model_name, len(parts))
            resp = self.model.generate_content(parts)
            return {"answer": resp.text}
        except Exception as e:
            log.warning("Primary Gemini call failed: %s", e)
            try:
                # Attempt to log available models to help debugging
                available = [m.name for m in genai.list_models()]
                log.info("Available models: %s", available)
                
                # Try a very safe fallback if available
                fallback_name = "models/gemini-1.5-flash" if "models/gemini-1.5-flash" in available else available[0]
                log.info("Trying fallback to %s", fallback_name)
                
                fallback = genai.GenerativeModel(fallback_name)
                resp = fallback.generate_content(parts)
                return {"answer": resp.text}
            except Exception as e2:
                log.error("Gemini critical failure: %s", e2)
                return {"error": f"AI Error: {str(e2)}"}

    def stream_response(self, transcript):
        yield f"data: {json.dumps({'error':'Gemini streaming not used here'})}\n\n"

PROVIDERS = {"openai": OpenAIProvider, "google": GoogleProvider}

def get_provider(name):
    cls = PROVIDERS.get(name)
    if not cls: raise ValueError("Invalid provider")
    return cls()

# ---------- Flask ----------
app = Flask(__name__)
# Restrict CORS specifically to chrome extensions and localhost
CORS(app, resources={r"/*":{"origins":["chrome-extension://*","http://localhost:*","http://127.0.0.1:*"]}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5000 per hour", "200 per minute"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        expected_key = os.environ.get("SCRIBE_API_KEY")
        if expected_key:
            token = request.headers.get("X-API-Key")
            if token != expected_key:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

import mimetypes
from flask import send_file

db_error = None
try:
    import urllib.parse
    import ssl
    import pg8000.dbapi
    pg8000.dbapi.paramstyle = 'format'
except Exception as e:
    db_error = f"Import error: {e}"

def get_db_connection():
    global db_error
    if db_error: return None
    db_url = os.environ.get("POSTGRES_URL")
    if not db_url: return None
    try:
        parsed = urllib.parse.urlparse(db_url)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            database=parsed.path[1:],
            port=parsed.port or 5432,
            ssl_context=context
        )
    except Exception as e:
        db_error = f"Connect error: {e}"
        return None

# Initialize table
try:
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scribe_sessions (
                id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(255),
                transcript TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.close()
        conn.commit()
        conn.close()
except:
    pass

@app.get("/favicon.ico")
@app.get("/favicon.png")
def favicon():
    ico = os.path.join(os.path.dirname(__file__), "favicon.png")
    if os.path.exists(ico):
        return send_file(ico, mimetype="image/png")
    return "", 204

@app.get("/")
@app.get("/api")
@app.get("/api/")
@app.get("/api/index")
@app.get("/api/index.py")
def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scribe — AI Interview &amp; Meeting Assistant</title>
<meta name="description" content="Scribe listens to any conversation, captures every word in real-time, and turns it into summaries and answers — stealth-mode AI for interviews and meetings.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

:root{
  --bg:#07090f;
  --bg1:#0a0d16;
  --bg2:#0e1220;
  --bg3:#131828;
  --surface:rgba(255,255,255,0.04);
  --surface-md:rgba(255,255,255,0.07);
  --surface-hi:rgba(255,255,255,0.11);
  --accent:#7c6bff;
  --accent2:#a78bfa;
  --accent3:#c4b5fd;
  --accent-glow:rgba(124,107,255,0.28);
  --accent-glow-sm:rgba(124,107,255,0.12);
  --danger:#ff5b6a;
  --success:#34d399;
  --text:#eef0f8;
  --text2:#9099b5;
  --text3:#5a6280;
  --text4:#3d4462;
  --border:rgba(255,255,255,0.08);
  --border2:rgba(255,255,255,0.13);
  --border-accent:rgba(124,107,255,0.35);
  --r:10px;
  --r2:16px;
  --r3:22px;
  --r4:28px;
  --ease:cubic-bezier(0.16,1,0.3,1);
  --ease-spring:cubic-bezier(0.34,1.56,0.64,1);
}

html{scroll-behavior:smooth}
body{
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
  font-size:16px;line-height:1.6;
  overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ── Ambient background ── */
body::before{
  content:'';position:fixed;inset:0;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%,rgba(124,107,255,0.12) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%,rgba(99,102,241,0.1) 0%,transparent 60%),
    linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);
  pointer-events:none;z-index:0;
}

/* dot grid */
.g-grid{
  position:fixed;inset:0;z-index:0;
  background-image:radial-gradient(circle,rgba(255,255,255,0.04) 1px,transparent 1px);
  background-size:32px 32px;
  mask-image:radial-gradient(ellipse 100% 80% at 50% 0%,black 0%,transparent 80%);
  pointer-events:none;
}

/* ── NAV ── */
nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 44px;height:60px;
  background:rgba(7,9,15,0.82);
  backdrop-filter:blur(28px) saturate(160%);
  -webkit-backdrop-filter:blur(28px) saturate(160%);
  border-bottom:1px solid var(--border);
}
nav::after{
  content:'';position:absolute;bottom:-1px;left:15%;right:15%;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  opacity:0.4;
}

.logo{
  display:flex;align-items:center;gap:10px;
  text-decoration:none;font-size:15px;font-weight:700;
  color:var(--text);letter-spacing:-0.3px;
}
.logo-mark{
  width:30px;height:30px;
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 14px var(--accent-glow);
  flex-shrink:0;position:relative;overflow:hidden;
}
.logo-mark::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,0.25) 0%,transparent 60%);
}
.logo-mark svg{width:15px;height:15px;color:white;position:relative;z-index:1}
.logo-text{
  background:linear-gradient(135deg,var(--text) 0%,var(--accent3) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}

.nav-mid{display:flex;gap:32px;list-style:none}
.nav-mid a{color:var(--text3);text-decoration:none;font-size:14px;font-weight:500;transition:color .2s}
.nav-mid a:hover{color:var(--text)}

.nav-badge{
  display:flex;align-items:center;gap:7px;
  background:rgba(52,211,153,0.08);
  border:1px solid rgba(52,211,153,0.22);
  padding:6px 14px;border-radius:100px;
  font-family:'JetBrains Mono',monospace;font-size:11px;
  color:var(--success);letter-spacing:1px;font-weight:500;
}
.nav-badge-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--success);
  animation:pulseDot 2s ease-in-out infinite;
}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

/* ── HERO ── */
#hero{
  position:relative;z-index:1;
  min-height:100vh;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  text-align:center;
  padding:120px 24px 80px;
  overflow:hidden;
}
.hero-glow{
  position:absolute;top:-80px;left:50%;transform:translateX(-50%);
  width:900px;height:600px;
  background:radial-gradient(ellipse,rgba(124,107,255,0.12) 0%,transparent 65%);
  pointer-events:none;
}

.eyebrow{
  display:inline-flex;align-items:center;gap:10px;
  font-family:'JetBrains Mono',monospace;font-size:11px;
  letter-spacing:2.5px;text-transform:uppercase;color:var(--accent);
  margin-bottom:32px;
  animation:fadeUp .6s .05s both;
}
.eyebrow-line{width:36px;height:1px;background:linear-gradient(90deg,transparent,var(--accent))}
.eyebrow-line.r{background:linear-gradient(90deg,var(--accent),transparent)}

h1{
  font-size:clamp(48px,8vw,96px);
  font-weight:800;line-height:1.0;
  letter-spacing:-4px;color:var(--text);
  max-width:900px;
  animation:fadeUp .7s .12s both;
}
h1 .grad{
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 50%,#818cf8 100%);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;
}

.hero-sub{
  font-size:18px;font-weight:400;color:var(--text2);
  max-width:520px;margin:24px auto 0;line-height:1.78;
  animation:fadeUp .7s .22s both;
}

.cta-row{
  display:flex;gap:14px;align-items:center;
  justify-content:center;margin-top:44px;
  animation:fadeUp .7s .32s both;
  flex-wrap:wrap;
}
.btn-primary{
  display:inline-flex;align-items:center;gap:8px;
  background:linear-gradient(135deg,var(--accent) 0%,#6366f1 100%);
  color:white;font-size:15px;font-weight:700;
  padding:14px 30px;border-radius:100px;
  text-decoration:none;letter-spacing:-.1px;
  transition:transform .2s var(--ease),box-shadow .2s;
  box-shadow:0 4px 20px var(--accent-glow);
}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 36px rgba(124,107,255,0.45)}
.btn-secondary{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--surface);color:var(--text2);
  font-size:15px;font-weight:500;padding:14px 24px;border-radius:100px;
  text-decoration:none;border:1px solid var(--border2);
  transition:color .2s,border-color .2s,background .2s;
}
.btn-secondary:hover{color:var(--text);border-color:var(--border-accent);background:var(--surface-md)}

/* ── Pillars ── */
.pillars{
  display:flex;gap:1px;
  background:var(--border);
  border:1px solid var(--border2);
  border-radius:var(--r2);overflow:hidden;
  max-width:780px;width:100%;
  margin:56px auto 0;
  animation:fadeUp .7s .42s both;
}
.pillar{
  flex:1;padding:22px 20px;
  background:var(--bg1);text-align:center;
  transition:background .2s;
}
.pillar:hover{background:var(--bg2)}
.pillar-icon{
  width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,rgba(124,107,255,0.18) 0%,rgba(167,139,250,0.1) 100%);
  border:1px solid rgba(124,107,255,0.2);
  display:flex;align-items:center;justify-content:center;
  margin:0 auto 12px;font-size:18px;
}
.pillar-title{font-size:14px;font-weight:700;color:var(--text);margin-bottom:5px;letter-spacing:-.2px}
.pillar-desc{font-size:12px;color:var(--text3);line-height:1.6}

/* ── Mock Window ── */
.mock-wrap{
  position:relative;z-index:1;
  width:100%;max-width:920px;
  margin:64px auto 0;
  animation:fadeUp .7s .5s both;
}
.mock{
  background:rgba(14,18,32,0.7);
  backdrop-filter:blur(20px);
  border:1px solid var(--border2);
  border-radius:var(--r3);overflow:hidden;
  box-shadow:0 32px 80px rgba(0,0,0,0.6),0 0 0 1px rgba(255,255,255,0.05) inset;
}
.mock-bar{
  display:flex;align-items:center;gap:7px;
  padding:13px 18px;
  background:rgba(10,13,22,0.8);
  border-bottom:1px solid var(--border);
}
.mdot{width:11px;height:11px;border-radius:50%}
.mdot.r{background:#ff5f56}.mdot.y{background:#ffbd2e}.mdot.g{background:#27c93f}
.mock-title{
  margin:0 auto;font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--text4);letter-spacing:.5px;
}
.mock-body{display:grid;grid-template-columns:1fr 1px 1fr;min-height:280px}
.mock-pane{padding:24px 22px}
.mock-label{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  letter-spacing:2px;text-transform:uppercase;
  color:var(--text4);margin-bottom:14px;
  display:flex;align-items:center;gap:7px;
}
.mock-label-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--success);
  animation:pulseDot 1.8s ease-in-out infinite;
}
.mock-div{background:var(--border);width:1px}
.transcript-line{font-size:13px;color:var(--text2);line-height:1.85;margin-bottom:5px}
.transcript-line .who{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  letter-spacing:1px;color:var(--accent3);
  text-transform:uppercase;margin-right:7px;font-weight:600;
}
.tcursor{
  display:inline-block;width:2px;height:14px;
  background:var(--accent);vertical-align:middle;margin-left:2px;
  animation:blink 1s step-end infinite;border-radius:1px;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

.summary-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:14px 16px;margin-bottom:10px;font-size:13px;
}
.sum-head{
  font-family:'JetBrains Mono',monospace;font-size:9px;
  letter-spacing:1.8px;text-transform:uppercase;
  color:var(--text4);margin-bottom:9px;
}
.sum-item{
  display:flex;align-items:flex-start;gap:8px;
  color:var(--text2);padding:3px 0;line-height:1.55;
}
.sum-bullet{color:var(--accent);font-size:12px;margin-top:1px}

/* ── SECTION LABEL ── */
.sec-label{
  font-family:'JetBrains Mono',monospace;font-size:11px;
  letter-spacing:2.5px;text-transform:uppercase;color:var(--accent);
  margin-bottom:14px;display:flex;align-items:center;gap:10px;
}
.sec-label::before{content:'';width:14px;height:1px;background:var(--accent)}
.sec-h{
  font-size:clamp(30px,4vw,48px);font-weight:800;
  letter-spacing:-2px;color:var(--text);
  line-height:1.05;max-width:580px;margin-bottom:56px;
}

/* ── FEATURES BENTO ── */
#features{
  position:relative;z-index:1;
  padding:120px 48px;max-width:1200px;margin:0 auto;
}
.feat-grid{
  display:grid;
  grid-template-columns:repeat(12,1fr);
  gap:2px;
  background:var(--border);
  border:1px solid var(--border2);
  border-radius:var(--r3);
  overflow:hidden;
}
.fc{
  background:var(--bg1);padding:40px 36px;
  position:relative;overflow:hidden;
  transition:background .25s;
}
.fc:hover{background:var(--bg2)}
.fc::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(500px circle at var(--mx,50%) var(--my,50%),rgba(124,107,255,0.07),transparent 40%);
  opacity:0;transition:opacity .3s;
}
.fc:hover::before{opacity:1}
.f7{grid-column:span 7}.f5{grid-column:span 5}
.f6{grid-column:span 6}.f4{grid-column:span 4}
.f8{grid-column:span 8}.f12{grid-column:span 12}

.fc-icon{
  width:48px;height:48px;border-radius:12px;
  background:linear-gradient(135deg,rgba(124,107,255,0.2) 0%,rgba(167,139,250,0.1) 100%);
  border:1px solid rgba(124,107,255,0.2);
  display:flex;align-items:center;justify-content:center;
  font-size:22px;margin-bottom:20px;
  box-shadow:0 4px 16px rgba(124,107,255,0.15);
}
.fc-title{
  font-size:21px;font-weight:700;color:var(--text);
  letter-spacing:-.5px;margin-bottom:10px;line-height:1.2;
}
.fc-desc{
  font-size:14px;color:var(--text2);
  line-height:1.75;font-weight:400;max-width:360px;
}

/* feature demo insets */
.demo{
  margin-top:24px;background:var(--bg);
  border:1px solid var(--border2);border-radius:var(--r);
  overflow:hidden;font-size:13px;
}
.demo-bar{
  display:flex;align-items:center;gap:6px;
  padding:9px 14px;background:var(--bg1);
  border-bottom:1px solid var(--border);
}
.demo-title{
  margin:0 auto;font-family:'JetBrains Mono',monospace;
  font-size:10px;color:var(--text4);letter-spacing:.5px;
}
.demo-body{padding:16px 18px;line-height:1.8}
.d-row{color:var(--text2);margin-bottom:2px}
.d-hi{color:var(--text);font-weight:600}
.d-muted{color:var(--text4)}
.d-accent{color:var(--accent3)}
.d-green{color:var(--success)}

/* big stat */
.big-n{
  font-family:'JetBrains Mono',monospace;
  font-size:72px;font-weight:500;
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;
  letter-spacing:-4px;line-height:1;margin-top:24px;
}
.big-n-sub{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--text4);
  letter-spacing:1.5px;text-transform:uppercase;margin-top:7px;
}
.mini-bars{
  display:flex;align-items:flex-end;gap:5px;
  height:52px;margin-top:20px;
}
.mb{
  flex:1;border-radius:3px 3px 0 0;
  background:linear-gradient(180deg,rgba(124,107,255,0.4) 0%,rgba(124,107,255,0.1) 100%);
  border-top:1px solid rgba(124,107,255,0.5);
}

/* Waveform inside feature card */
.wave-row{
  display:flex;align-items:flex-end;gap:3px;
  height:52px;justify-content:center;margin:20px 0;
}
.wb{
  width:5px;border-radius:3px 3px 0 0;
  background:linear-gradient(180deg,var(--accent) 0%,var(--accent2) 100%);
  opacity:.7;animation:wv 1.2s ease-in-out infinite;
}
@keyframes wv{0%,100%{transform:scaleY(.15)}50%{transform:scaleY(1)}}

/* ── HOW IT WORKS ── */
#how{
  position:relative;z-index:1;
  background:rgba(10,13,22,0.5);
  border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);
  padding:120px 48px;
}
.how-inner{
  max-width:1200px;margin:0 auto;
  display:grid;grid-template-columns:1fr 1fr;
  gap:80px;align-items:start;
}
.steps{margin-top:48px}
.step{
  display:flex;gap:22px;padding:24px 0;
  border-bottom:1px solid var(--border);
  cursor:pointer;transition:padding .2s;
}
.step:last-child{border-bottom:none}
.step:hover{padding-left:4px}
.step-n{
  font-family:'JetBrains Mono',monospace;font-size:12px;
  color:var(--text4);padding-top:3px;
  flex-shrink:0;width:24px;transition:color .2s;
}
.step.on .step-n{color:var(--accent)}
.step-title{
  font-size:18px;font-weight:700;
  color:var(--text3);letter-spacing:-.3px;
  margin-bottom:0;transition:color .2s;
}
.step.on .step-title{color:var(--text)}
.step-desc{
  font-size:14px;color:var(--text4);line-height:1.7;
  font-weight:400;max-height:0;overflow:hidden;
  transition:max-height .4s var(--ease),opacity .3s,margin .3s;
  opacity:0;
}
.step.on .step-desc{max-height:120px;opacity:1;color:var(--text2);margin-top:9px}

.how-visual{
  position:sticky;top:90px;
  background:rgba(14,18,32,0.8);
  backdrop-filter:blur(20px);
  border:1px solid var(--border2);
  border-radius:var(--r3);padding:28px;min-height:360px;
  display:flex;align-items:center;
  box-shadow:0 16px 48px rgba(0,0,0,0.4);
}
.how-visual::before{
  content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);
}
.vis{display:none;width:100%;animation:fin .35s var(--ease)}
.vis.on{display:block}
@keyframes fin{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* ── CTA ── */
#cta{
  position:relative;z-index:1;
  text-align:center;padding:140px 48px;overflow:hidden;
}
#cta::before{
  content:'';position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  width:800px;height:600px;
  background:radial-gradient(ellipse,rgba(124,107,255,0.1) 0%,transparent 65%);
  pointer-events:none;
}
.cta-h{
  font-size:clamp(40px,7vw,84px);font-weight:800;
  letter-spacing:-3px;color:var(--text);line-height:1.0;
  margin-bottom:24px;
}
.cta-h .grad{
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 50%,#818cf8 100%);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;
}
.cta-sub{
  font-size:18px;color:var(--text2);font-weight:400;
  max-width:460px;margin:0 auto 48px;line-height:1.75;
}

/* ── FOOTER ── */
footer{
  position:relative;z-index:1;
  border-top:1px solid var(--border);
  padding:32px 48px;
  display:flex;align-items:center;justify-content:space-between;
}
.foot-logo{
  font-size:14px;font-weight:700;color:var(--text3);
  text-decoration:none;letter-spacing:-.2px;
}
.foot-links{display:flex;gap:24px;list-style:none}
.foot-links a{
  color:var(--text4);font-size:13px;
  text-decoration:none;transition:color .2s;
}
.foot-links a:hover{color:var(--text3)}
.foot-copy{font-size:12px;color:var(--text4)}

/* ── Scroll reveal ── */
.rv{opacity:0;transform:translateY(28px);transition:opacity .7s var(--ease),transform .7s var(--ease)}
.rv.in{opacity:1;transform:none}

/* ── Keyframes ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(124,107,255,0.3);border-radius:2px}

/* ── Responsive ── */
@media(max-width:900px){
  nav{padding:0 20px}
  .nav-mid{display:none}
  h1{letter-spacing:-2px}
  #features{padding:80px 20px}
  .feat-grid{grid-template-columns:1fr}
  .f7,.f5,.f6,.f4,.f8,.f12{grid-column:span 1}
  .how-inner{grid-template-columns:1fr;gap:40px}
  .how-visual{position:relative;top:0}
  #how{padding:80px 20px}
  footer{flex-direction:column;gap:16px;text-align:center;padding:32px 20px}
  .pillars{flex-direction:column}
}
</style>
</head>
<body>

<div class="g-grid"></div>

<!-- NAV -->
<nav>
  <a class="logo" href="#">
    <div class="logo-mark">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 20h9"/>
        <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4 12.5-12.5z"/>
      </svg>
    </div>
    <span class="logo-text">Scribe</span>
  </a>
  <ul class="nav-mid">
    <li><a href="#features">Features</a></li>
    <li><a href="#how">How it works</a></li>
    <li><a href="https://github.com/Shubh-Pandey99/scribe" target="_blank">GitHub</a></li>
  </ul>
  <div class="nav-badge">
    <span class="nav-badge-dot"></span>
    Live
  </div>
</nav>

<!-- HERO -->
<section id="hero">
  <div class="hero-glow"></div>

  <div class="eyebrow">
    <span class="eyebrow-line"></span>
    Stealth AI Assistant
    <span class="eyebrow-line r"></span>
  </div>

  <h1>
    Your secret weapon<br>
    <span class="grad">for every meeting.</span>
  </h1>

  <p class="hero-sub">
    Scribe listens live, transcribes in real-time, and answers your questions — silently running in your browser sidepanel. No one knows it&apos;s there.
  </p>

  <div class="cta-row">
    <a href="https://github.com/Shubh-Pandey99/scribe" target="_blank" class="btn-primary">
      <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 00-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0020 4.77 5.07 5.07 0 0019.91 1S18.73.65 16 2.48a13.38 13.38 0 00-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 005 4.77a5.44 5.44 0 00-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 009 18.13V22"/></svg>
      Add to Chrome — free
    </a>
    <a href="#how" class="btn-secondary">
      See how it works
      <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
    </a>
  </div>

  <!-- Three pillars -->
  <div class="pillars">
    <div class="pillar">
      <div class="pillar-icon">🎙</div>
      <div class="pillar-title">Capture</div>
      <div class="pillar-desc">Every word, live — tab audio or mic</div>
    </div>
    <div class="pillar">
      <div class="pillar-icon">✨</div>
      <div class="pillar-title">Analyse</div>
      <div class="pillar-desc">Summaries, screenshots, AI answers</div>
    </div>
    <div class="pillar">
      <div class="pillar-icon">👁</div>
      <div class="pillar-title">Stealth</div>
      <div class="pillar-desc">Invisible until you need it</div>
    </div>
  </div>

  <!-- Mock window -->
  <div class="mock-wrap rv">
    <div class="mock">
      <div class="mock-bar">
        <div class="mdot r"></div><div class="mdot y"></div><div class="mdot g"></div>
        <span class="mock-title">Scribe — Interview · Live</span>
      </div>
      <div class="mock-body">
        <!-- Transcript pane -->
        <div class="mock-pane">
          <div class="mock-label">
            <span class="mock-label-dot"></span>
            Listening now
          </div>
          <div class="transcript-line"><span class="who">Q</span>Tell me about a time you had to optimise a slow query in production.</div>
          <div class="transcript-line"><span class="who">You</span>Sure — at my last role we had a JOIN across three tables hitting 4 seconds on mobile.<span class="tcursor"></span></div>
        </div>
        <div class="mock-div"></div>
        <!-- AI pane -->
        <div class="mock-pane">
          <div class="mock-label">AI Answer</div>
          <div class="summary-card">
            <div class="sum-head">Key points to cover</div>
            <div class="sum-item"><span class="sum-bullet">›</span>Identify the bottleneck (EXPLAIN ANALYZE)</div>
            <div class="sum-item"><span class="sum-bullet">›</span>Add composite index on join columns</div>
            <div class="sum-item"><span class="sum-bullet">›</span>Result: 4s → 120ms (33× improvement)</div>
          </div>
          <div class="summary-card">
            <div class="sum-head">Follow-up answers ready</div>
            <div class="sum-item"><span class="sum-bullet">›</span>Index type chosen and why</div>
            <div class="sum-item"><span class="sum-bullet">›</span>Impact on write performance</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- FEATURES -->
<section id="features">
  <div class="sec-label rv">What Scribe does</div>
  <h2 class="sec-h rv">Everything you need.<br>Nothing you don&apos;t.</h2>

  <div class="feat-grid rv">

    <!-- 1: Transcription -->
    <div class="fc f7">
      <div class="fc-icon">🎙</div>
      <div class="fc-title">Hear every word.</div>
      <div class="fc-desc">
        Open Scribe, hit record, and talk. It listens to your mic or any browser tab — Zoom, Google Meet, Teams, video calls — and shows you a live transcript as you speak.
      </div>
      <div class="demo" style="margin-top:24px">
        <div class="demo-bar">
          <div class="mdot r"></div><div class="mdot y"></div><div class="mdot g"></div>
          <span class="demo-title">Live Transcript</span>
        </div>
        <div class="demo-body">
          <div class="d-row"><span class="d-accent">Q</span> &nbsp;Walk me through your system design experience.</div>
          <div class="d-row"><span class="d-accent">You</span> &nbsp;I&apos;ve designed distributed systems at scale including...<span class="tcursor"></span></div>
        </div>
      </div>
    </div>

    <!-- 2: Stealth -->
    <div class="fc f5">
      <div class="fc-icon">👁</div>
      <div class="fc-title">Stay invisible.</div>
      <div class="fc-desc">
        One click activates stealth mode — the panel fades and blurs. Hover to reveal. No one on your call ever sees it. Your secret co-pilot.
      </div>
      <div style="margin-top:24px;padding:20px;background:var(--bg);border:1px solid var(--border2);border-radius:var(--r);text-align:center">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text4);letter-spacing:1.5px;margin-bottom:12px">STEALTH MODE</div>
        <div style="filter:blur(4px);opacity:0.4;font-size:13px;color:var(--text2);line-height:1.7">The answer is to use a Redis-based<br>rate limiter with token bucket...</div>
        <div style="margin-top:10px;font-size:11px;color:var(--accent3)">← hover to reveal</div>
      </div>
    </div>

    <!-- 3: Speed -->
    <div class="fc f4">
      <div class="fc-icon">⚡</div>
      <div class="fc-title">Instant.</div>
      <div class="fc-desc">Answers in under 3 seconds. You&apos;re never left waiting while the interviewer moves on.</div>
      <div class="big-n">&lt;3<span style="font-size:.4em;-webkit-text-fill-color:var(--text3)">s</span></div>
      <div class="big-n-sub">avg response time</div>
    </div>

    <!-- 4: Ask anything -->
    <div class="fc f8">
      <div class="fc-icon">💬</div>
      <div class="fc-title">Ask anything about what was said.</div>
      <div class="fc-desc">
        Missed something? Ask Scribe. &ldquo;What was the follow-up question?&rdquo; &ldquo;Who&apos;s handling the design?&rdquo; It reads everything it heard and gives you a direct answer instantly.
      </div>
      <div class="demo" style="margin-top:20px">
        <div class="demo-bar">
          <div class="mdot r"></div><div class="mdot y"></div><div class="mdot g"></div>
          <span class="demo-title">Ask Scribe</span>
        </div>
        <div class="demo-body">
          <div class="d-row d-muted">Ask anything about this meeting...</div>
          <div class="d-row" style="margin-top:8px"><span class="d-accent">You:</span> What system design patterns should I mention?</div>
          <div class="d-row"><span style="color:var(--accent3)">Scribe:</span> <span class="d-hi">CQRS, Event Sourcing, Saga pattern</span> — all relevant here</div>
        </div>
      </div>
    </div>

    <!-- 5: Screen snap -->
    <div class="fc f6">
      <div class="fc-icon">📷</div>
      <div class="fc-title">See what&apos;s on screen too.</div>
      <div class="fc-desc">
        Sharing a coding challenge or diagram? Screenshot it and paste. Scribe reads the image alongside your conversation and gives context-aware answers about what&apos;s shown.
      </div>
    </div>

    <!-- 6: History -->
    <div class="fc f6">
      <div class="fc-icon">🗂</div>
      <div class="fc-title">Every session, saved.</div>
      <div class="fc-desc">
        All your sessions are saved locally and synced to the cloud. Go back to any interview, re-read the transcript, get a fresh summary, or ask new questions about old conversations.
      </div>
    </div>

  </div>
</section>

<!-- HOW IT WORKS -->
<div id="how">
  <div class="how-inner">
    <div>
      <div class="sec-label rv">How it works</div>
      <h2 class="sec-h rv" style="margin-bottom:0">Three steps.<br>That&apos;s it.</h2>
      <div class="steps rv">
        <div class="step on" data-v="0">
          <div class="step-n">01</div>
          <div>
            <div class="step-title">Add Scribe to Chrome</div>
            <div class="step-desc">Clone the repo and load it as an unpacked extension. Open the sidebar on any tab and paste your API URL — takes about 60 seconds total.</div>
          </div>
        </div>
        <div class="step" data-v="1">
          <div class="step-n">02</div>
          <div>
            <div class="step-title">Hit record on any call</div>
            <div class="step-desc">Start a Zoom, Meet, or any audio tab and press Record. Scribe starts listening immediately — the transcript appears live as people speak.</div>
          </div>
        </div>
        <div class="step" data-v="2">
          <div class="step-n">03</div>
          <div>
            <div class="step-title">Get answers instantly</div>
            <div class="step-desc">Ask questions via the command bar or hit Summarise. AI answers appear in under 3 seconds, formatted and scannable while you&apos;re speaking.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="how-visual rv">
      <!-- Visual 0: setup -->
      <div class="vis on" id="v0">
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;width:100%">
          <div style="color:var(--text4);margin-bottom:18px;letter-spacing:.5px">⚙ Setup — 60 seconds</div>
          <div style="background:var(--bg1);border:1px solid var(--border2);border-radius:var(--r);overflow:hidden">
            <div style="padding:14px 18px;border-bottom:1px solid var(--border);font-size:11px;color:var(--text4)">Settings</div>
            <div style="padding:20px 18px;display:flex;flex-direction:column;gap:14px">
              <div>
                <div style="font-size:10px;color:var(--text4);letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">API URL</div>
                <div style="background:var(--bg);border:1px solid var(--border-accent);border-radius:8px;padding:9px 12px;color:var(--accent3);font-size:12px">https://scribe-extension.vercel.app</div>
              </div>
              <div style="background:linear-gradient(135deg,var(--accent) 0%,#6366f1 100%);color:white;text-align:center;padding:11px;border-radius:8px;font-weight:700;font-size:13px;box-shadow:0 4px 14px rgba(124,107,255,0.4)">Save &amp; Connect ✓</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Visual 1: recording -->
      <div class="vis" id="v1">
        <div style="width:100%">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text4);letter-spacing:1px;text-transform:uppercase">Recording</div>
            <div style="color:var(--danger);display:flex;align-items:center;gap:6px;font-size:11px;font-family:'JetBrains Mono',monospace">
              <span style="width:6px;height:6px;border-radius:50%;background:var(--danger);display:inline-block;animation:pulseDot 1.4s infinite"></span>
              Live
            </div>
          </div>
          <div style="display:flex;justify-content:center;gap:3px;height:48px;align-items:flex-end;margin-bottom:16px">
            <div class="wb" style="animation-delay:0s"></div>
            <div class="wb" style="animation-delay:.12s"></div>
            <div class="wb" style="animation-delay:.24s"></div>
            <div class="wb" style="animation-delay:.12s"></div>
            <div class="wb" style="animation-delay:0s"></div>
          </div>
          <div style="background:var(--bg1);border:1px solid var(--border2);border-radius:var(--r);padding:16px 18px;font-size:13px;color:var(--text2);line-height:1.8">
            &ldquo;Let&apos;s confirm — the system needs to handle 10k RPS at P99 under 100ms.&rdquo;<span class="tcursor"></span>
          </div>
        </div>
      </div>

      <!-- Visual 2: answer -->
      <div class="vis" id="v2">
        <div style="width:100%;font-size:13px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text4);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:14px">AI Answer</div>
          <div style="background:var(--bg1);border:1px solid var(--border2);border-radius:var(--r);margin-bottom:10px">
            <div style="padding:12px 16px;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text4);letter-spacing:1.5px;text-transform:uppercase">Architecture to mention</div>
            <div style="padding:14px 16px">
              <div style="color:var(--text2);margin-bottom:6px">› <span style="color:var(--text);font-weight:600">Read replicas</span> + connection pooling (PgBouncer)</div>
              <div style="color:var(--text2);margin-bottom:6px">› <span style="color:var(--text);font-weight:600">Redis cache</span> for hot paths (TTL: 5min)</div>
              <div style="color:var(--text2)">› <span style="color:var(--text);font-weight:600">CDN edge</span> for static + API gateway rate limiting</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- CTA -->
<section id="cta">
  <h2 class="cta-h rv">
    Your next interview,<br>
    <span class="grad">fully prepared.</span>
  </h2>
  <p class="cta-sub rv">Free, open source, and running in your browser. No data leaves your machine unless you choose it.</p>
  <div class="cta-row rv">
    <a href="https://github.com/Shubh-Pandey99/scribe" target="_blank" class="btn-primary" style="font-size:16px;padding:16px 36px">
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 00-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0020 4.77 5.07 5.07 0 0019.91 1S18.73.65 16 2.48a13.38 13.38 0 00-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 005 4.77a5.44 5.44 0 00-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 009 18.13V22"/></svg>
      Get Scribe on GitHub
    </a>
  </div>
</section>

<!-- FOOTER -->
<footer>
  <a class="foot-logo" href="#">Scribe</a>
  <ul class="foot-links">
    <li><a href="https://github.com/Shubh-Pandey99/scribe" target="_blank">GitHub</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#how">How it works</a></li>
  </ul>
  <div class="foot-copy">MIT License &middot; Open Source</div>
</footer>

<script>
// How it works steps
document.querySelectorAll('.step').forEach(s=>{
  s.addEventListener('click',()=>{
    document.querySelectorAll('.step').forEach(x=>x.classList.remove('on'))
    s.classList.add('on')
    const id='v'+s.dataset.v
    document.querySelectorAll('.vis').forEach(v=>v.classList.remove('on'))
    document.getElementById(id)?.classList.add('on')
  })
})

// Scroll reveal
const obs=new IntersectionObserver(es=>{
  es.forEach(e=>{ if(e.isIntersecting){e.target.classList.add('in');obs.unobserve(e.target)} })
},{threshold:.08,rootMargin:'0px 0px -30px 0px'})
document.querySelectorAll('.rv').forEach(el=>obs.observe(el))

// Feature card mouse glow
document.querySelectorAll('.fc').forEach(fc=>{
  fc.addEventListener('mousemove',e=>{
    const r=fc.getBoundingClientRect()
    fc.style.setProperty('--mx',(e.clientX-r.left)+'px')
    fc.style.setProperty('--my',(e.clientY-r.top)+'px')
  })
})
</script>
</body>
</html>
""", 200, {"Content-Type": "text/html"}

@app.get("/health")
def health(): return jsonify({"status":"ok", "db_error": db_error}), 200

# -------- Sessions API (Postgres) --------
@app.get("/api/sessions")
@require_api_key
def get_sessions():
    conn = get_db_connection()
    if not conn: return jsonify({"error": f"No database attached. {db_error}"}), 503
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, title, transcript, created_at FROM scribe_sessions ORDER BY created_at DESC LIMIT 50")
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        
        # Convert datetime to string
        for r in rows:
            if r.get('created_at'): r['started_at'] = r['created_at'].isoformat()
        return jsonify(rows), 200
    except Exception as e:
        log.exception("get sessions failed")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.post("/api/sessions")
@require_api_key
def save_session():
    data = request.get_json(force=True) or {}
    sid = data.get("id")
    title = data.get("title", "Untitled session")
    transcript = data.get("transcript", "")
    if not sid: return jsonify({"error": "Missing id"}), 400
    
    conn = get_db_connection()
    if not conn: return jsonify({"error": "No database attached"}), 503
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO scribe_sessions (id, title, transcript) 
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET 
              title = EXCLUDED.title, 
              transcript = EXCLUDED.transcript
        """, (sid, title, transcript))
        cur.close()
        conn.commit()
        return jsonify({"status": "saved", "id": sid}), 200
    except Exception as e:
        log.exception("save session failed")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.delete("/api/sessions/<session_id>")
@require_api_key
def delete_session(session_id):
    conn = get_db_connection()
    if not conn: return jsonify({"error": "No database attached"}), 503
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM scribe_sessions WHERE id = %s", (session_id,))
        cur.close()
        conn.commit()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.post("/api/answer")
@require_api_key
@limiter.limit("30 per minute")
def answer():
    try:
        data = request.get_json(force=True) or {}
        provider_name = data.get("provider","google")
        transcript = data.get("transcript")
        image_url = data.get("imageUrl")
        image_base64 = data.get("imageBase64")
        image_array = data.get("imageArray")

        provider = get_provider(provider_name)
        result = provider.get_response(transcript=transcript, image_url=image_url, image_base64=image_base64, image_array=image_array)
        if "error" in result: return jsonify(result), 400
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.exception("answer failed")
        return jsonify({"error":"Server error"}), 500

# -------- Simple chunked STT (Whisper-1) --------
# Accepts audioBase64 (webm/opus) chunks and returns incremental text.
from werkzeug.utils import secure_filename

@app.post("/api/transcribe")
@require_api_key
@limiter.limit("30 per minute")
def transcribe():
    try:
        data = request.get_json(force=True) or {}
        audio_b64 = data.get("audioBase64")
        mime = data.get("mimeType","audio/webm")
        session_id = secure_filename(data.get("sessionId","default"))
        previous_text = data.get("previousText", "")
        if not audio_b64: return jsonify({"error":"No audioBase64"}), 400

        # decode to temp file
        if "," in audio_b64:
            header, encoded = audio_b64.split(",",1)
        else:
            encoded = audio_b64
        
        # Fix base64 padding (browsers sometimes omit trailing '=')
        encoded = encoded.strip()
        padding = 4 - len(encoded) % 4
        if padding != 4:
            encoded += '=' * padding
        
        buf = base64.b64decode(encoded)
        log.info("Received audio chunk: %d bytes, mime=%s", len(buf), mime)
        
        if len(buf) < 100:
            return jsonify({"text": "", "method": "skip", "debug": "chunk too small"}), 200
        
        suffix = ".webm" if "webm" in mime else ".ogg" if "ogg" in mime else ".wav" if "wav" in mime else ".mp4" if "mp4" in mime else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(buf)
            tmp_path = f.name

        text = ""
        method = "none"
        debug_info = ""
        
        # PRIMARY: Groq Whisper (free, fast, reliable)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                groq_client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                with open(tmp_path, "rb") as fp:
                    tr = groq_client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=fp, prompt=previous_text)
                text = getattr(tr, "text", "").strip()
                method = "groq-whisper"
                debug_info = f"groq returned {len(text)} chars"
                log.info("Groq Whisper result: '%s'", text[:100])
            except Exception as e:
                err_str = str(e)
                debug_info = f"groq error: {err_str[:100]}"
                # If rate limited, set method so frontend shows what happened
                if "429" in err_str or "rate" in err_str.lower():
                    method = "groq-ratelimit"
                    log.warning("Groq rate limited, trying fallbacks")
                else:
                    log.warning("Groq failed: %s, trying OpenAI fallback", e)
        else:
            debug_info = "no GROQ_API_KEY"
        
        # Fallback 1: OpenAI Whisper
        if not text:
            oai_key = os.getenv("OPENAI_API_KEY")
            if oai_key:
                try:
                    client = OpenAI(api_key=oai_key)
                    with open(tmp_path, "rb") as fp:
                        tr = client.audio.transcriptions.create(model="whisper-1", file=fp, prompt=previous_text)
                    text = getattr(tr, "text", "").strip()
                    method = "whisper"
                    debug_info += f" | whisper returned {len(text)} chars"
                    log.info("Whisper result: '%s'", text[:100])
                except Exception as e:
                    debug_info += f" | whisper error: {str(e)[:100]}"
                    log.warning("Whisper failed: %s, trying Gemini fallback", e)
        
        # Fallback 2: Google Gemini for audio transcription (with retry for 429)
        if not text:
            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key:
                genai.configure(api_key=google_key, transport="rest")
                audio_mime = mime if mime else "audio/webm"
                audio_part = {
                    "inline_data": {
                        "mime_type": audio_mime,
                        "data": encoded
                    }
                }
                prompt = (
                    "Transcribe this audio exactly. Output ONLY the spoken words, nothing else. "
                    "If there is music but no speech, output just the word MUSIC. If completely silent, output SILENT. "
                    f"Previous context for smooth stitching: '{previous_text[-200:]}'"
                )
                
                # Try with retry + model fallback for rate limits
                models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "models/gemini-1.5-flash"]
                for model_name in models_to_try:
                    for attempt in range(3):
                        try:
                            model = genai.GenerativeModel(model_name)
                            resp = model.generate_content([prompt, audio_part])
                            raw = resp.text.strip() if resp.text else ""
                            method = f"gemini({model_name})"
                            debug_info += f" | {model_name} returned: '{raw[:60]}'"
                            if raw and raw not in ("MUSIC", "SILENT", ""):
                                text = raw
                            log.info("Gemini STT [%s] result: '%s'", model_name, raw[:100])
                            break
                        except Exception as e2:
                            err_str = str(e2)
                            if "429" in err_str and attempt < 2:
                                time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
                                continue
                            debug_info += f" | {model_name} err: {err_str[:60]}"
                            log.warning("Gemini STT [%s] attempt %d failed: %s", model_name, attempt, e2)
                            break
                    if text:
                        break
            else:
                debug_info += " | no GOOGLE_API_KEY"
        
        try: os.remove(tmp_path)
        except: pass
        
        return jsonify({"text": text, "method": method, "debug": debug_info})
    except Exception as e:
        log.exception("transcribe failed")
        return jsonify({"error": f"Transcription error: {str(e)}"}), 500



if __name__ == "__main__":
    host = os.getenv("HOST","0.0.0.0")
    port = int(os.getenv("PORT",5055))
    debug = os.getenv("DEBUG","true").lower()=="true"
    log.info("Starting AnswerAI API on %s:%s (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
